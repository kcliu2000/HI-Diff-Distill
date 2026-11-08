from pathlib import Path
from PIL import Image
import hashlib


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

EXPECTED_COUNT = 2103

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def collect_images(root):
    if not root.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root}"
        )

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    mapping = {}
    duplicates = {}

    for path in paths:
        key = path.name

        if key in mapping:
            duplicates.setdefault(
                key,
                [mapping[key]],
            ).append(path)
        else:
            mapping[key] = path

    return mapping, duplicates


def print_duplicates(label, duplicates):
    if not duplicates:
        return

    print(f"\nDuplicate {label} filenames:")

    for name, paths in list(
        duplicates.items()
    )[:10]:
        print(f"  {name}")

        for path in paths:
            print(f"    {path}")


def image_info(path):
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
        }


def short_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()[:16]


blur, blur_duplicates = collect_images(
    BLUR_ROOT
)

gt, gt_duplicates = collect_images(
    GT_ROOT
)

teacher, teacher_duplicates = collect_images(
    TEACHER_ROOT
)

print("Dataset roots")
print(f"  Blur:    {BLUR_ROOT}")
print(f"  GT:      {GT_ROOT}")
print(f"  Teacher: {TEACHER_ROOT.resolve()}")

print()
print("File counts")
print(f"  Blur:    {len(blur)}")
print(f"  GT:      {len(gt)}")
print(f"  Teacher: {len(teacher)}")

print_duplicates(
    "Blur",
    blur_duplicates,
)

print_duplicates(
    "GT",
    gt_duplicates,
)

print_duplicates(
    "Teacher",
    teacher_duplicates,
)

blur_names = set(blur)
gt_names = set(gt)
teacher_names = set(teacher)

missing_gt = sorted(
    blur_names - gt_names
)

missing_teacher = sorted(
    blur_names - teacher_names
)

extra_gt = sorted(
    gt_names - blur_names
)

extra_teacher = sorted(
    teacher_names - blur_names
)

print()
print("Filename comparison")
print(f"  Missing GT:      {len(missing_gt)}")
print(
    f"  Missing Teacher: "
    f"{len(missing_teacher)}"
)
print(f"  Extra GT:        {len(extra_gt)}")
print(
    f"  Extra Teacher:   "
    f"{len(extra_teacher)}"
)

if missing_gt:
    print(
        "  First missing GT:",
        missing_gt[:10],
    )

if missing_teacher:
    print(
        "  First missing Teacher:",
        missing_teacher[:10],
    )

if extra_gt:
    print(
        "  First extra GT:",
        extra_gt[:10],
    )

if extra_teacher:
    print(
        "  First extra Teacher:",
        extra_teacher[:10],
    )

common_names = sorted(
    blur_names
    & gt_names
    & teacher_names
)

shape_errors = []
mode_warnings = []
read_errors = []

for index, name in enumerate(
    common_names,
    start=1,
):
    try:
        blur_info = image_info(blur[name])
        gt_info = image_info(gt[name])
        teacher_info = image_info(
            teacher[name]
        )
    except Exception as error:
        read_errors.append(
            (
                name,
                repr(error),
            )
        )
        continue

    sizes = {
        blur_info["size"],
        gt_info["size"],
        teacher_info["size"],
    }

    if len(sizes) != 1:
        shape_errors.append(
            {
                "name": name,
                "blur": blur_info["size"],
                "gt": gt_info["size"],
                "teacher": teacher_info["size"],
            }
        )

    modes = {
        blur_info["mode"],
        gt_info["mode"],
        teacher_info["mode"],
    }

    if modes != {"RGB"}:
        mode_warnings.append(
            {
                "name": name,
                "blur": blur_info["mode"],
                "gt": gt_info["mode"],
                "teacher": teacher_info["mode"],
            }
        )

    if (
        index % 250 == 0
        or index == len(common_names)
    ):
        print(
            f"Checked: {index}/"
            f"{len(common_names)}"
        )

print()
print("Image validation")
print(
    f"  Shape errors:    "
    f"{len(shape_errors)}"
)
print(
    f"  Mode warnings:   "
    f"{len(mode_warnings)}"
)
print(
    f"  Read errors:     "
    f"{len(read_errors)}"
)

if shape_errors:
    print("\nFirst shape errors:")

    for item in shape_errors[:10]:
        print(" ", item)

if mode_warnings:
    print("\nFirst mode warnings:")

    for item in mode_warnings[:10]:
        print(" ", item)

if read_errors:
    print("\nFirst read errors:")

    for item in read_errors[:10]:
        print(" ", item)

sample_names = common_names[:3]

if sample_names:
    print()
    print("Sample triplets")

    for name in sample_names:
        print(f"  {name}")
        print(f"    Blur:    {blur[name]}")
        print(f"    GT:      {gt[name]}")
        print(
            f"    Teacher: {teacher[name]}"
        )
        print(
            f"    Teacher SHA256: "
            f"{short_hash(teacher[name])}"
        )

valid = all(
    [
        len(blur) == EXPECTED_COUNT,
        len(gt) == EXPECTED_COUNT,
        len(teacher) == EXPECTED_COUNT,
        not blur_duplicates,
        not gt_duplicates,
        not teacher_duplicates,
        not missing_gt,
        not missing_teacher,
        not extra_gt,
        not extra_teacher,
        not shape_errors,
        not read_errors,
    ]
)

print()
print("=" * 60)

if valid:
    print("KD DATASET VERIFICATION: PASS")
else:
    print("KD DATASET VERIFICATION: FAIL")

print("=" * 60)

if not valid:
    raise SystemExit(1)
