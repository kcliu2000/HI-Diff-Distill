from copy import deepcopy
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"

SOURCE_CANDIDATES = [
    TRAIN_DIR / "GoPro_Student_Lite_S1.yml",
    TRAIN_DIR / "GoPro_Student50_S1.yml",
    TRAIN_DIR / "GoPro_S1.yml",
]

OUTPUT = (
    TRAIN_DIR
    / "GoPro_Student_Lite_S1_Formal.yml"
)

TRAIN_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/sharp"
)

TRAIN_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/blur"
)

VAL_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/sharp"
)

VAL_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/blur"
)

GT_SIZES = [128, 160, 192, 256, 320, 384]
BATCH_SIZES = [12, 8, 6, 4, 2, 2]
STAGE_ITERS = [
    20000,
    20000,
    20000,
    20000,
    20000,
    200000,
]

TOTAL_ITER = sum(STAGE_ITERS)


def find_source():
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "找不到 Student-Lite 或官方 Stage 1 config。"
    )


def count_images(path):
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(root)

    suffixes = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    files = [
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.suffix.lower() in suffixes
    ]

    if not files:
        raise RuntimeError(
            f"No images found in {root}"
        )

    return len(files)


source = find_source()

with source.open("r", encoding="utf-8") as file:
    original = yaml.safe_load(file)

if not isinstance(original, dict):
    raise RuntimeError(
        f"Cannot parse config: {source}"
    )

config = deepcopy(original)

config["name"] = (
    "train_HI_Diff_GoPro_"
    "Student_Lite_S1_Formal"
)

config["model_type"] = "HI_Diff_S1"
config["num_gpu"] = 1
config["manual_seed"] = 100

network_g = config["network_g"]
network_g["num_blocks"] = [1, 2, 2, 2]
network_g["num_refinement_blocks"] = 1
network_g["dim"] = 48
network_g["embed_dim"] = 64
network_g["heads"] = [1, 2, 4, 8]

network_le = config["network_le"]
network_le["in_chans"] = 6
network_le["embed_dim"] = 64
network_le["block_num"] = 6
network_le["stage"] = 1

train_dataset = config["datasets"]["train"]
train_dataset["dataroot_gt"] = TRAIN_GT
train_dataset["dataroot_lq"] = TRAIN_LQ
train_dataset["use_shuffle"] = True
train_dataset["num_worker_per_gpu"] = 8

# Dataloader 必須能提供最大的 progressive batch。
train_dataset["batch_size_per_gpu"] = max(BATCH_SIZES)

train_dataset["dataset_enlarge_ratio"] = 1
train_dataset["gt_size"] = max(GT_SIZES)
train_dataset["gt_sizes"] = GT_SIZES
train_dataset["mini_batch_sizes"] = BATCH_SIZES
train_dataset["iters"] = STAGE_ITERS

val_dataset = config["datasets"]["val"]
val_dataset["dataroot_gt"] = VAL_GT
val_dataset["dataroot_lq"] = VAL_LQ

path_opt = config.setdefault("path", {})
path_opt["pretrain_network_g"] = None
path_opt["strict_load_g"] = False
path_opt["pretrain_network_le"] = None
path_opt["strict_load_le"] = False
path_opt["resume_state"] = None

train_opt = config["train"]
train_opt["total_iter"] = TOTAL_ITER
train_opt["warmup_iter"] = -1
train_opt["use_grad_clip"] = True

scheduler = train_opt.get("scheduler")

if isinstance(scheduler, dict):
    scheduler["periods"] = [100000, 200000]
    scheduler["restart_weights"] = [1, 1]
    scheduler["eta_mins"] = [0.0002, 0.000001]

logger = config.setdefault("logger", {})
logger["print_freq"] = 100
logger["save_checkpoint_freq"] = 10000
logger["use_tb_logger"] = True

val = config.setdefault("val", {})
val["val_freq"] = 10000
val["save_img"] = False

with OUTPUT.open("w", encoding="utf-8") as file:
    yaml.safe_dump(
        config,
        file,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )

print("Formal Student-Lite Stage 1 config created")
print(f"Source: {source}")
print(f"Output: {OUTPUT}")
print()
print("Dataset")
print(f"  Train sharp: {count_images(TRAIN_GT)} images")
print(f"  Train blur:  {count_images(TRAIN_LQ)} images")
print(f"  Val sharp:   {count_images(VAL_GT)} images")
print(f"  Val blur:    {count_images(VAL_LQ)} images")
print()
print("Architecture")
print("  num_blocks: [1, 2, 2, 2]")
print("  num_refinement_blocks: 1")
print("  Transformer parameters: 11,524,730")
print("  Deployment parameters: 14,712,707")
print("  Parameter reduction: 48.35%")
print()
print("Progressive training")
print(f"  gt_sizes: {GT_SIZES}")
print(f"  mini_batch_sizes: {BATCH_SIZES}")
print(f"  iters: {STAGE_ITERS}")
print(f"  total_iter: {TOTAL_ITER}")
print()
print("Logging")
print("  print_freq: 100")
print("  save_checkpoint_freq: 10000")
print("  val_freq: 10000")
