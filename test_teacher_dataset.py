import copy
import torch

import hi_diff.data
from basicsr.data import build_dataset


base_opt = {
    "name": "KDTrainSetTest",
    "type": "PairedImageTeacherDataset",
    "dataroot_gt": (
        "/home/m11302124/MIMO-UNet-Wavelet/"
        "dataset/GOPRO/train/sharp"
    ),
    "dataroot_lq": (
        "/home/m11302124/MIMO-UNet-Wavelet/"
        "dataset/GOPRO/train/blur"
    ),
    "dataroot_teacher": (
        "/home/m11302124/MIMO-UNet-Wavelet/"
        "dataset/GOPRO/train/teacher_hidiff"
    ),
    "filename_tmpl": "{}",
    "io_backend": {
        "type": "disk",
    },
    "gt_size": 128,
    "use_hflip": True,
    "use_rot": True,
    "phase": "train",
    "scale": 1,
}


dataset = build_dataset(
    copy.deepcopy(base_opt)
)

print("Dataset length:", len(dataset))

if len(dataset) != 2103:
    raise RuntimeError(
        f"Expected 2103 samples, "
        f"got {len(dataset)}"
    )


for index in (0, 1, 100, 1000, 2102):
    sample = dataset[index]

    lq = sample["lq"]
    gt = sample["gt"]
    teacher = sample["teacher"]

    print()
    print("Index:", index)
    print("  LQ:", tuple(lq.shape))
    print("  GT:", tuple(gt.shape))
    print(
        "  Teacher:",
        tuple(teacher.shape),
    )
    print(
        "  LQ range:",
        float(lq.min()),
        float(lq.max()),
    )
    print(
        "  GT range:",
        float(gt.min()),
        float(gt.max()),
    )
    print(
        "  Teacher range:",
        float(teacher.min()),
        float(teacher.max()),
    )

    expected_shape = (3, 128, 128)

    if tuple(lq.shape) != expected_shape:
        raise RuntimeError(
            f"Invalid LQ shape: {lq.shape}"
        )

    if tuple(gt.shape) != expected_shape:
        raise RuntimeError(
            f"Invalid GT shape: {gt.shape}"
        )

    if (
        tuple(teacher.shape)
        != expected_shape
    ):
        raise RuntimeError(
            "Invalid teacher shape: "
            f"{teacher.shape}"
        )

    for name, tensor in (
        ("lq", lq),
        ("gt", gt),
        ("teacher", teacher),
    ):
        if not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                f"{name} contains NaN/Inf"
            )

        if (
            tensor.min() < 0
            or tensor.max() > 1
        ):
            raise RuntimeError(
                f"{name} is outside [0, 1]"
            )


print()
print("PairedImageTeacherDataset test: PASS")
