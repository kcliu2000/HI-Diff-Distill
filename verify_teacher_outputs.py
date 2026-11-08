from pathlib import Path
import random

import numpy as np
from PIL import Image


BLUR_ROOT = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/blur"
)

GT_ROOT = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/sharp"
)

TEACHER_ROOT = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/teacher_hidiff"
)

random.seed(100)

teacher_files = sorted(
    path
    for path in TEACHER_ROOT.rglob("*.png")
)

if len(teacher_files) != 2103:
    raise RuntimeError(
        f"Expected 2103 teacher files, "
        f"found {len(teacher_files)}"
    )

samples = random.sample(
    teacher_files,
    k=min(20, len(teacher_files)),
)

print(
    "name,"
    "mae_teacher_blur,"
    "mae_teacher_gt,"
    "mae_blur_gt"
)

identical_to_blur = []
identical_to_gt = []

for teacher_path in samples:
    name = teacher_path.name

    blur_path = BLUR_ROOT / name
    gt_path = GT_ROOT / name

    if not blur_path.exists():
        matches = list(
            BLUR_ROOT.rglob(name)
        )

        if not matches:
            raise FileNotFoundError(
                f"Missing blur: {name}"
            )

        blur_path = matches[0]

    if not gt_path.exists():
        matches = list(
            GT_ROOT.rglob(name)
        )

        if not matches:
            raise FileNotFoundError(
                f"Missing GT: {name}"
            )

        gt_path = matches[0]

    teacher = np.asarray(
        Image.open(
            teacher_path
        ).convert("RGB"),
        dtype=np.float32,
    )

    blur = np.asarray(
        Image.open(
            blur_path
        ).convert("RGB"),
        dtype=np.float32,
    )

    gt = np.asarray(
        Image.open(
            gt_path
        ).convert("RGB"),
        dtype=np.float32,
    )

    mae_teacher_blur = np.mean(
        np.abs(teacher - blur)
    )

    mae_teacher_gt = np.mean(
        np.abs(teacher - gt)
    )

    mae_blur_gt = np.mean(
        np.abs(blur - gt)
    )

    print(
        f"{name},"
        f"{mae_teacher_blur:.4f},"
        f"{mae_teacher_gt:.4f},"
        f"{mae_blur_gt:.4f}"
    )

    if np.array_equal(
        teacher,
        blur,
    ):
        identical_to_blur.append(name)

    if np.array_equal(
        teacher,
        gt,
    ):
        identical_to_gt.append(name)

print()
print(
    "Identical to blur:",
    len(identical_to_blur),
)

print(
    "Identical to GT:",
    len(identical_to_gt),
)

if identical_to_blur:
    print(
        "Unexpected blur copies:",
        identical_to_blur,
    )

if identical_to_gt:
    print(
        "Unexpected GT copies:",
        identical_to_gt,
    )

if identical_to_blur or identical_to_gt:
    raise SystemExit(1)

print("Teacher output sample check: PASS")
