from copy import deepcopy
from pathlib import Path
import os
import shutil
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"

SOURCE_CONFIG = (
    TRAIN_DIR / "GoPro_Student_Lite_S1.yml"
)

TRAIN_GT = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/sharp"
)

TRAIN_LQ = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/blur"
)

TEST_GT = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/sharp"
)

TEST_LQ = Path(
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/blur"
)

SMOKE_ROOT = ROOT / "datasets" / "benchmark_val"
SMOKE_GT = SMOKE_ROOT / "sharp"
SMOKE_LQ = SMOKE_ROOT / "blur"

BATCH_SIZES = [8, 4, 2]
BENCHMARK_ITERATIONS = 10
PATCH_SIZE = 384


def list_images(root):
    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
    )


def verify_dataset(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 不存在：{path}"
        )

    images = list_images(path)

    if not images:
        raise RuntimeError(
            f"{description} 沒有影像：{path}"
        )

    print(
        f"{description}: {len(images)} images"
    )

    return images


def find_matching_pair(lq_images, gt_root):
    for lq_path in lq_images:
        relative = lq_path.relative_to(TEST_LQ)
        gt_path = gt_root / relative

        if gt_path.exists():
            return lq_path, gt_path

        same_name = list(
            gt_root.rglob(lq_path.name)
        )

        if same_name:
            return lq_path, same_name[0]

    raise RuntimeError(
        "找不到檔名或相對路徑相符的測試影像。"
    )


def create_validation_pair():
    lq_images = verify_dataset(
        TEST_LQ,
        "GoPro test blur",
    )

    verify_dataset(
        TEST_GT,
        "GoPro test sharp",
    )

    lq_path, gt_path = find_matching_pair(
        lq_images,
        TEST_GT,
    )

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)

    SMOKE_LQ.mkdir(
        parents=True,
        exist_ok=True,
    )

    SMOKE_GT.mkdir(
        parents=True,
        exist_ok=True,
    )

    lq_link = SMOKE_LQ / lq_path.name
    gt_link = SMOKE_GT / gt_path.name

    os.symlink(
        lq_path.resolve(),
        lq_link,
    )

    os.symlink(
        gt_path.resolve(),
        gt_link,
    )

    print()
    print("Benchmark validation pair")
    print(f"  Blur:  {lq_path}")
    print(f"  Sharp: {gt_path}")


def update_scheduler(config):
    train_opt = config["train"]
    scheduler = train_opt.get("scheduler")

    if isinstance(scheduler, dict):
        periods = scheduler.get("periods")

        if isinstance(periods, list):
            scheduler["periods"] = [
                BENCHMARK_ITERATIONS
            ]

        restart_weights = scheduler.get(
            "restart_weights"
        )

        if isinstance(
            restart_weights,
            list,
        ):
            scheduler["restart_weights"] = [
                1
            ]

        eta_mins = scheduler.get(
            "eta_mins"
        )

        if isinstance(eta_mins, list):
            scheduler["eta_mins"] = [
                1e-6
            ]


def create_config(base, batch_size):
    config = deepcopy(base)

    name = (
        "bench_HI_Diff_GoPro_"
        f"Student_Lite_S1_p384_b{batch_size}"
    )

    config["name"] = name
    config["num_gpu"] = 1
    config["manual_seed"] = 100

    network_g = config["network_g"]
    network_g["num_blocks"] = [
        1, 2, 2, 2
    ]
    network_g[
        "num_refinement_blocks"
    ] = 1

    train_dataset = (
        config["datasets"]["train"]
    )

    train_dataset["dataroot_gt"] = str(
        TRAIN_GT
    )

    train_dataset["dataroot_lq"] = str(
        TRAIN_LQ
    )

    train_dataset[
        "num_worker_per_gpu"
    ] = 4

    train_dataset[
        "batch_size_per_gpu"
    ] = batch_size

    train_dataset["gt_size"] = PATCH_SIZE

    train_dataset["gt_sizes"] = [
        PATCH_SIZE
    ]

    train_dataset[
        "mini_batch_sizes"
    ] = [
        batch_size
    ]

    train_dataset["iters"] = [
        BENCHMARK_ITERATIONS
    ]

    val_dataset = config["datasets"]["val"]

    val_dataset["dataroot_gt"] = str(
        SMOKE_GT
    )

    val_dataset["dataroot_lq"] = str(
        SMOKE_LQ
    )

    train_opt = config["train"]
    train_opt["total_iter"] = (
        BENCHMARK_ITERATIONS
    )

    update_scheduler(config)

    path_opt = config.setdefault(
        "path", {}
    )

    path_opt["pretrain_network_g"] = None
    path_opt["strict_load_g"] = False

    path_opt["pretrain_network_le"] = None
    path_opt["strict_load_le"] = False

    path_opt["resume_state"] = None

    val_opt = config.setdefault(
        "val", {}
    )

    val_opt["val_freq"] = 1000
    val_opt["save_img"] = False

    logger_opt = config.setdefault(
        "logger", {}
    )

    logger_opt["print_freq"] = 1

    logger_opt[
        "save_checkpoint_freq"
    ] = 1000

    output = (
        TRAIN_DIR
        / (
            "GoPro_Student_Lite_S1_"
            f"p384_b{batch_size}_bench.yml"
        )
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    print(f"Created: {output}")


if not SOURCE_CONFIG.exists():
    raise FileNotFoundError(
        "找不到正式 Student-Lite Stage 1 "
        f"設定：{SOURCE_CONFIG}"
    )

verify_dataset(
    TRAIN_GT,
    "GoPro train sharp",
)

verify_dataset(
    TRAIN_LQ,
    "GoPro train blur",
)

create_validation_pair()

with SOURCE_CONFIG.open(
    "r",
    encoding="utf-8",
) as file:
    base = yaml.safe_load(file)

if not isinstance(base, dict):
    raise RuntimeError(
        f"無法解析：{SOURCE_CONFIG}"
    )

for batch_size in BATCH_SIZES:
    create_config(
        base,
        batch_size,
    )

print()
print("Batch benchmark configs ready")
print("  Patch size: 384 x 384")
print("  Batch sizes: 8, 4, 2")
print("  Iterations per test: 10")
print("  Validation images: 1")
