from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from hi_diff.models.HI_Diff_S2_OutputKD_model import (
    HI_Diff_S2_OutputKD,
)


def haar_dwt_2d(x):
    """Differentiable one-level 2D Haar wavelet transform.

    Args:
        x: Tensor with shape [B, C, H, W].

    Returns:
        ll, lh, hl, hh:
            Four tensors with shape [B, C, H/2, W/2].
    """
    if x.ndim != 4:
        raise ValueError(
            "Haar DWT expects a 4D tensor, "
            f"but received shape {tuple(x.shape)}."
        )

    height = x.shape[-2]
    width = x.shape[-1]

    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Haar DWT requires even spatial dimensions, "
            f"but received H={height}, W={width}."
        )

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    # Orthonormal Haar scaling.
    ll = (
        x00
        + x01
        + x10
        + x11
    ) * 0.5

    lh = (
        -x00
        + x01
        - x10
        + x11
    ) * 0.5

    hl = (
        -x00
        - x01
        + x10
        + x11
    ) * 0.5

    hh = (
        x00
        - x01
        - x10
        + x11
    ) * 0.5

    return ll, lh, hl, hh


@MODEL_REGISTRY.register()
class HI_Diff_S2_OutputWaveletKD(
    HI_Diff_S2_OutputKD
):
    """HI-Diff Stage 2 with Output KD and Wavelet KD.

    L_total =
        L_deblur
        + L_prior
        + alpha * L_output_kd_raw
        + beta * (
            w_ll * L_ll
            + w_lh * L_lh
            + w_hl * L_hl
            + w_hh * L_hh
        )

    Output KD is initialized by the parent OutputKD model.
    Teacher outputs are pre-generated RGB images loaded by
    PairedImageTeacherDataset.
    """

    def init_training_settings(self):
        super().init_training_settings()

        wavelet_opt = self.opt["train"].get(
            "wavelet_kd_opt"
        )

        if wavelet_opt is None:
            raise ValueError(
                "train.wavelet_kd_opt is required."
            )

        self.wavelet_beta = float(
            wavelet_opt.get(
                "loss_weight",
                0.5,
            )
        )

        self.wavelet_reduction = (
            wavelet_opt.get(
                "reduction",
                "mean",
            )
        )

        if self.wavelet_reduction not in (
            "mean",
            "sum",
        ):
            raise ValueError(
                "wavelet_kd_opt.reduction must be "
                "'mean' or 'sum', but received "
                f"{self.wavelet_reduction}."
            )

        self.wavelet_ll_weight = float(
            wavelet_opt.get(
                "ll_weight",
                1.0,
            )
        )

        self.wavelet_lh_weight = float(
            wavelet_opt.get(
                "lh_weight",
                2.0,
            )
        )

        self.wavelet_hl_weight = float(
            wavelet_opt.get(
                "hl_weight",
                2.0,
            )
        )

        self.wavelet_hh_weight = float(
            wavelet_opt.get(
                "hh_weight",
                2.0,
            )
        )

        for name, value in (
            ("loss_weight", self.wavelet_beta),
            ("ll_weight", self.wavelet_ll_weight),
            ("lh_weight", self.wavelet_lh_weight),
            ("hl_weight", self.wavelet_hl_weight),
            ("hh_weight", self.wavelet_hh_weight),
        ):
            if value < 0:
                raise ValueError(
                    f"wavelet_kd_opt.{name} must be "
                    f"non-negative, but received {value}."
                )

    def optimize_parameters(
        self,
        current_iter,
        noise=None,
    ):
        if self.teacher is None:
            raise KeyError(
                "Output + Wavelet KD training requires "
                "a teacher tensor. Use "
                "PairedImageTeacherDataset for the "
                "training dataset."
            )

        # Stage 1 latent encoder is fixed during Stage 2.
        self.net_le.eval()

        for parameter in self.net_le.parameters():
            parameter.requires_grad = False

        self.optimizer_total.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():
            prior_z = self.net_le(
                self.lq,
                self.gt,
            )

        if self.apply_ldm:
            prior, _ = self.diffusion(
                self.lq,
                prior_z,
            )
        else:
            prior_condition = self.net_le_dm(
                self.lq
            )

            timesteps = self.opt[
                "diffusion_schedule"
            ]["timesteps"]

            if noise is None:
                noise = torch.randn_like(
                    prior_z
                )

            prior_noisy = self.q_sample(
                x_start=prior_z,
                sqrt_alpha_cumprod=(
                    self.alphas_cumprod[
                        timesteps - 1
                    ]
                ),
                noise=noise,
            )

            prior = (
                self.p_sample_loop_wo_variance(
                    prior_condition,
                    prior_noisy,
                )
            )

        self.output = self.net_g(
            self.lq,
            prior,
        )

        if self.output.shape != self.gt.shape:
            raise RuntimeError(
                "Student output and GT shapes differ: "
                f"{tuple(self.output.shape)} vs "
                f"{tuple(self.gt.shape)}"
            )

        if self.output.shape != self.teacher.shape:
            raise RuntimeError(
                "Student output and teacher output "
                "shapes differ: "
                f"{tuple(self.output.shape)} vs "
                f"{tuple(self.teacher.shape)}"
            )

        loss_dict = OrderedDict()

        # Ground-truth reconstruction loss.
        l_pix = self.cri_pix(
            self.output,
            self.gt,
        )

        # Diffusion latent-prior reconstruction loss.
        l_pix_diff = self.cri_pix_diff(
            prior_z,
            prior,
        )

        # Raw RGB Output KD for monitoring.
        l_output_kd_raw = F.l1_loss(
            self.output,
            self.teacher,
            reduction="mean",
        )

        # Weighted RGB Output KD. Alpha is included in
        # output_kd_opt.loss_weight.
        l_output_kd = self.cri_output_kd(
            self.output,
            self.teacher,
        )

        # Student output remains attached to the graph.
        (
            student_ll,
            student_lh,
            student_hl,
            student_hh,
        ) = haar_dwt_2d(self.output)

        # Teacher output is an offline fixed target.
        with torch.no_grad():
            (
                teacher_ll,
                teacher_lh,
                teacher_hl,
                teacher_hh,
            ) = haar_dwt_2d(self.teacher)

        expected_shape = (
            self.output.shape[0],
            self.output.shape[1],
            self.output.shape[2] // 2,
            self.output.shape[3] // 2,
        )

        for band_name, band_tensor in (
            ("student_ll", student_ll),
            ("student_lh", student_lh),
            ("student_hl", student_hl),
            ("student_hh", student_hh),
            ("teacher_ll", teacher_ll),
            ("teacher_lh", teacher_lh),
            ("teacher_hl", teacher_hl),
            ("teacher_hh", teacher_hh),
        ):
            if band_tensor.shape != expected_shape:
                raise RuntimeError(
                    f"{band_name} has shape "
                    f"{tuple(band_tensor.shape)}, "
                    f"expected {expected_shape}."
                )

        l_wavelet_ll = F.l1_loss(
            student_ll,
            teacher_ll,
            reduction=self.wavelet_reduction,
        )

        l_wavelet_lh = F.l1_loss(
            student_lh,
            teacher_lh,
            reduction=self.wavelet_reduction,
        )

        l_wavelet_hl = F.l1_loss(
            student_hl,
            teacher_hl,
            reduction=self.wavelet_reduction,
        )

        l_wavelet_hh = F.l1_loss(
            student_hh,
            teacher_hh,
            reduction=self.wavelet_reduction,
        )

        # Internal sub-band weighting.
        l_wavelet_kd_raw = (
            self.wavelet_ll_weight
            * l_wavelet_ll
            + self.wavelet_lh_weight
            * l_wavelet_lh
            + self.wavelet_hl_weight
            * l_wavelet_hl
            + self.wavelet_hh_weight
            * l_wavelet_hh
        )

        # Overall Wavelet KD coefficient beta.
        l_wavelet_kd = (
            self.wavelet_beta
            * l_wavelet_kd_raw
        )

        l_total = (
            l_pix
            + l_pix_diff
            + l_output_kd
            + l_wavelet_kd
        )

        loss_dict["l_pix"] = l_pix
        loss_dict["l_pix_diff"] = l_pix_diff

        loss_dict[
            "l_output_kd_raw"
        ] = l_output_kd_raw.detach()

        loss_dict[
            "l_output_kd"
        ] = l_output_kd

        loss_dict[
            "l_wavelet_ll"
        ] = l_wavelet_ll.detach()

        loss_dict[
            "l_wavelet_lh"
        ] = l_wavelet_lh.detach()

        loss_dict[
            "l_wavelet_hl"
        ] = l_wavelet_hl.detach()

        loss_dict[
            "l_wavelet_hh"
        ] = l_wavelet_hh.detach()

        loss_dict[
            "l_wavelet_kd_raw"
        ] = l_wavelet_kd_raw.detach()

        loss_dict[
            "l_wavelet_kd"
        ] = l_wavelet_kd

        if self.cri_perceptual:
            l_percep, l_style = (
                self.cri_perceptual(
                    self.output,
                    self.gt,
                )
            )

            if l_percep is not None:
                l_total = (
                    l_total
                    + l_percep
                )
                loss_dict[
                    "l_percep"
                ] = l_percep

            if l_style is not None:
                l_total = (
                    l_total
                    + l_style
                )
                loss_dict[
                    "l_style"
                ] = l_style

        loss_dict["l_total"] = l_total

        if not torch.isfinite(l_total):
            raise FloatingPointError(
                "Non-finite loss detected at "
                f"iteration {current_iter}."
            )

        l_total.backward()

        # Gradient-path validation for smoke tests.
        if current_iter == 1:
            output_grad_detected = False

            for parameter in self.net_g.parameters():
                if (
                    parameter.grad is not None
                    and torch.isfinite(
                        parameter.grad
                    ).all()
                ):
                    output_grad_detected = True
                    break

            if not output_grad_detected:
                raise RuntimeError(
                    "No finite gradient was detected "
                    "in net_g after Wavelet KD backward."
                )

        if self.opt["train"].get(
            "use_grad_clip",
            False,
        ):
            if self.apply_ldm:
                parameters = (
                    list(self.net_g.parameters())
                    + list(
                        self.diffusion.parameters()
                    )
                )
            else:
                parameters = (
                    list(self.net_g.parameters())
                    + list(
                        self.net_le_dm.parameters()
                    )
                    + list(
                        self.net_d.parameters()
                    )
                )

            torch.nn.utils.clip_grad_norm_(
                parameters,
                0.01,
            )

        self.optimizer_total.step()

        self.log_dict = self.reduce_loss_dict(
            loss_dict
        )
