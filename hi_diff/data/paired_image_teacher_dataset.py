import os
import os.path as osp

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageTeacherDataset(data.Dataset):
    """Paired LQ, GT, and fixed teacher-output dataset.

    The LQ, GT, and teacher images must use identical filenames.
    The same crop and augmentation are applied to all three images.
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.file_client = None

        self.io_backend_opt = opt[
            "io_backend"
        ].copy()

        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.teacher_folder = opt[
            "dataroot_teacher"
        ]

        self.filename_tmpl = opt.get(
            "filename_tmpl",
            "{}",
        )

        self.mean = opt.get("mean")
        self.std = opt.get("std")

        paired_paths = paired_paths_from_folder(
            [
                self.lq_folder,
                self.gt_folder,
            ],
            ["lq", "gt"],
            self.filename_tmpl,
        )

        teacher_files = {}

        for path in sorted(
            osp.join(
                self.teacher_folder,
                filename,
            )
            for filename in os_listdir_images(
                self.teacher_folder
            )
        ):
            teacher_files[
                osp.basename(path)
            ] = path

        self.paths = []

        for item in paired_paths:
            filename = osp.basename(
                item["lq_path"]
            )

            if filename not in teacher_files:
                raise FileNotFoundError(
                    "Teacher output is missing for "
                    f"{filename}"
                )

            self.paths.append(
                {
                    "lq_path": item[
                        "lq_path"
                    ],
                    "gt_path": item[
                        "gt_path"
                    ],
                    "teacher_path": (
                        teacher_files[
                            filename
                        ]
                    ),
                }
            )

        if len(self.paths) != len(
            teacher_files
        ):
            paired_names = {
                osp.basename(
                    item["lq_path"]
                )
                for item in self.paths
            }

            extra_teacher = sorted(
                set(teacher_files)
                - paired_names
            )

            raise RuntimeError(
                "Teacher output count does not "
                "match paired LQ/GT count. "
                f"Paired: {len(self.paths)}, "
                f"Teacher: {len(teacher_files)}, "
                f"First extras: "
                f"{extra_teacher[:5]}"
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop(
                    "type"
                ),
                **self.io_backend_opt,
            )

        item = self.paths[index]

        lq_path = item["lq_path"]
        gt_path = item["gt_path"]
        teacher_path = item[
            "teacher_path"
        ]

        img_lq = imfrombytes(
            self.file_client.get(
                lq_path,
                "lq",
            ),
            float32=True,
        )

        img_gt = imfrombytes(
            self.file_client.get(
                gt_path,
                "gt",
            ),
            float32=True,
        )

        img_teacher = imfrombytes(
            self.file_client.get(
                teacher_path,
                "teacher",
            ),
            float32=True,
        )

        if self.opt["phase"] == "train":
            gt_size = self.opt[
                "gt_size"
            ]

            (
                img_gt,
                img_lq,
                img_teacher,
            ) = triple_random_crop(
                img_gt,
                img_lq,
                img_teacher,
                gt_size,
                lq_path,
            )

            (
                img_gt,
                img_lq,
                img_teacher,
            ) = augment(
                [
                    img_gt,
                    img_lq,
                    img_teacher,
                ],
                self.opt.get(
                    "use_hflip",
                    True,
                ),
                self.opt.get(
                    "use_rot",
                    True,
                ),
            )

        img_gt, img_lq, img_teacher = (
            img2tensor(
                [
                    img_gt,
                    img_lq,
                    img_teacher,
                ],
                bgr2rgb=True,
                float32=True,
            )
        )

        if (
            self.mean is not None
            or self.std is not None
        ):
            raise NotImplementedError(
                "Mean/std normalization is not "
                "enabled for KD datasets."
            )

        return {
            "lq": img_lq,
            "gt": img_gt,
            "teacher": img_teacher,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "teacher_path": teacher_path,
        }

    def __len__(self):
        return len(self.paths)


def os_listdir_images(root):
    suffixes = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    )

    return [
        filename
        for filename in os.listdir(root)
        if filename.lower().endswith(
            suffixes
        )
    ]


def triple_random_crop(
    img_gt,
    img_lq,
    img_teacher,
    gt_patch_size,
    lq_path,
):
    """Apply one crop window to GT, LQ, and teacher."""

    h_lq, w_lq = img_lq.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    h_teacher, w_teacher = (
        img_teacher.shape[:2]
    )

    if (
        h_lq != h_gt
        or w_lq != w_gt
        or h_lq != h_teacher
        or w_lq != w_teacher
    ):
        raise ValueError(
            "LQ, GT, and teacher dimensions "
            f"differ for {lq_path}: "
            f"LQ={img_lq.shape}, "
            f"GT={img_gt.shape}, "
            f"Teacher={img_teacher.shape}"
        )

    if (
        h_lq < gt_patch_size
        or w_lq < gt_patch_size
    ):
        raise ValueError(
            f"Image {lq_path} is smaller "
            f"than patch size "
            f"{gt_patch_size}: "
            f"{img_lq.shape}"
        )

    top = np.random.randint(
        0,
        h_lq - gt_patch_size + 1,
    )

    left = np.random.randint(
        0,
        w_lq - gt_patch_size + 1,
    )

    bottom = top + gt_patch_size
    right = left + gt_patch_size

    img_gt = img_gt[
        top:bottom,
        left:right,
        ...,
    ]

    img_lq = img_lq[
        top:bottom,
        left:right,
        ...,
    ]

    img_teacher = img_teacher[
        top:bottom,
        left:right,
        ...,
    ]

    return (
        img_gt,
        img_lq,
        img_teacher,
    )
