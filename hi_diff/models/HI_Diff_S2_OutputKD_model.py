from collections import OrderedDict

import torch

from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from hi_diff.models.HI_Diff_S2_model import HI_Diff_S2


@MODEL_REGISTRY.register()
class HI_Diff_S2_OutputKD(HI_Diff_S2):
    """HI-Diff Stage 2 with fixed RGB output KD.

    L_total = L_deblur + L_prior + alpha * L_output_kd

    The teacher output is pre-generated and loaded by the dataset.
    """

    def init_training_settings(self):
        super().init_training_settings()

        output_kd_opt = self.opt["train"].get(
            "output_kd_opt"
        )

        if output_kd_opt is None:
            raise ValueError(
                "train.output_kd_opt is required."
            )

        self.cri_output_kd = build_loss(
            output_kd_opt
        ).to(self.device)

    def feed_data(self, data):
        """Load training or validation data.

        Training batches contain LQ, GT, and teacher tensors.
        Validation batches contain only LQ and GT because the
        teacher target is not required during inference.
        """

        self.lq = data["lq"].to(
            self.device,
            non_blocking=True,
        )

        if "gt" in data:
            self.gt = data["gt"].to(
                self.device,
                non_blocking=True,
            )

        # Reset this value for every batch so a validation batch
        # cannot accidentally reuse a previous training target.
        self.teacher = None

        if "teacher" in data:
            self.teacher = data["teacher"].to(
                self.device,
                non_blocking=True,
            ).detach()

    def optimize_parameters(
        self,
        current_iter,
        noise=None,
    ):
        if self.teacher is None:
            raise KeyError(
                "Output KD training requires a teacher tensor. "
                "Use PairedImageTeacherDataset for the "
                "training dataset."
            )

        # Stage 1 latent encoder is fixed in Stage 2.
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

            prior = self.p_sample_loop_wo_variance(
                prior_condition,
                prior_noisy,
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

        if (
            self.output.shape
            != self.teacher.shape
        ):
            raise RuntimeError(
                "Student output and teacher output "
                "shapes differ: "
                f"{tuple(self.output.shape)} vs "
                f"{tuple(self.teacher.shape)}"
            )

        loss_dict = OrderedDict()

        # GT reconstruction loss.
        l_pix = self.cri_pix(
            self.output,
            self.gt,
        )

        # Diffusion latent-prior loss.
        l_pix_diff = self.cri_pix_diff(
            prior_z,
            prior,
        )

        # Raw RGB Output KD, used only for monitoring.
        l_output_kd_raw = torch.nn.functional.l1_loss(
            self.output,
            self.teacher,
            reduction="mean",
        )

        # Weighted Output KD. The alpha value is included
        # in output_kd_opt.loss_weight.
        l_output_kd = self.cri_output_kd(
            self.output,
            self.teacher,
        )

        l_total = (
            l_pix
            + l_pix_diff
            + l_output_kd
        )

        loss_dict["l_pix"] = l_pix
        loss_dict["l_pix_diff"] = l_pix_diff
        loss_dict[
            "l_output_kd_raw"
        ] = l_output_kd_raw.detach()
        loss_dict[
            "l_output_kd"
        ] = l_output_kd
        loss_dict["l_total"] = l_total

        if self.cri_perceptual:
            l_percep, l_style = (
                self.cri_perceptual(
                    self.output,
                    self.gt,
                )
            )

            if l_percep is not None:
                l_total = l_total + l_percep
                loss_dict["l_percep"] = (
                    l_percep
                )

            if l_style is not None:
                l_total = l_total + l_style
                loss_dict["l_style"] = (
                    l_style
                )

        if not torch.isfinite(l_total):
            raise FloatingPointError(
                "Non-finite loss detected at "
                f"iteration {current_iter}."
            )

        l_total.backward()

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
