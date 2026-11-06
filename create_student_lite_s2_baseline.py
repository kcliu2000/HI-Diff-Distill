from copy import deepcopy
from pathlib import Path
import re
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"
EXPERIMENTS_DIR = ROOT / "experiments"

SOURCE_CONFIG = TRAIN_DIR / "GoPro_S2.yml"

FORMAL_CONFIG = (
    TRAIN_DIR
    / "GoPro_Student_Lite_S2_Baseline.yml"
)

SMOKE_CONFIG = (
    TRAIN_DIR
    / "GoPro_Student_Lite_S2_Baseline_smoke.yml"
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


def get_iteration(path):
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def verify_directory(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} 不存在：{path}"
        )

    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
    ]

    if not files:
        raise RuntimeError(
            f"{description} 沒有影像：{path}"
        )

    print(
        f"{description}: {path} "
        f"({len(files)} files)"
    )


def find_stage1_experiment():
    candidates = []

    if not EXPERIMENTS_DIR.exists():
        raise FileNotFoundError(
            f"找不到 experiments：{EXPERIMENTS_DIR}"
        )

    for directory in EXPERIMENTS_DIR.iterdir():
        if not directory.is_dir():
            continue

        name = directory.name.lower()

        is_stage1 = (
            "s1" in name
            or "stage1" in name
        )

        is_student_lite = any(
            keyword in name
            for keyword in (
                "student_lite",
                "studentlite",
                "student50",
            )
        )

        is_smoke = "smoke" in name

        if (
            is_stage1
            and is_student_lite
            and is_smoke
        ):
            models_dir = directory / "models"

            if models_dir.exists():
                candidates.append(directory)

    if not candidates:
        existing = [
            path.name
            for path in EXPERIMENTS_DIR.iterdir()
            if path.is_dir()
        ]

        raise FileNotFoundError(
            "找不到 Student-Lite Stage 1 "
            "smoke experiment。\n"
            "Existing experiments:\n  "
            + "\n  ".join(sorted(existing))
        )

    candidates.sort(
        key=lambda path: path.stat().st_mtime
    )

    return candidates[-1]


def find_checkpoint(models_dir, prefix):
    numbered = []

    for path in models_dir.glob(
        f"{prefix}_*.pth"
    ):
        if "latest" in path.name.lower():
            continue

        iteration = get_iteration(path)

        if iteration >= 0:
            numbered.append(
                (iteration, path)
            )

    if numbered:
        numbered.sort(
            key=lambda item: item[0]
        )
        return numbered[-1][1]

    latest = models_dir / f"{prefix}_latest.pth"

    if latest.exists():
        return latest

    available = [
        path.name
        for path in models_dir.glob("*.pth")
    ]

    raise FileNotFoundError(
        f"找不到 {prefix} checkpoint："
        f"{models_dir}\n"
        f"Available files: {available}"
    )


def configure_datasets(config, smoke=False):
    datasets = config.setdefault(
        "datasets", {}
    )

    train_dataset = datasets.get("train")

    if not isinstance(
        train_dataset, dict
    ):
        raise KeyError(
            "設定檔缺少 datasets.train"
        )

    train_dataset["dataroot_gt"] = TRAIN_GT
    train_dataset["dataroot_lq"] = TRAIN_LQ
    train_dataset["num_worker_per_gpu"] = 2
    train_dataset["batch_size_per_gpu"] = 1

    if smoke:
        mini_batch_sizes = (
            train_dataset.get(
                "mini_batch_sizes"
            )
        )

        if isinstance(
            mini_batch_sizes, list
        ):
            train_dataset[
                "mini_batch_sizes"
            ] = [
                1
                for _ in mini_batch_sizes
            ]

        batch_sizes = train_dataset.get(
            "batch_sizes"
        )

        if isinstance(batch_sizes, list):
            train_dataset[
                "batch_sizes"
            ] = [
                1
                for _ in batch_sizes
            ]

    val_dataset = datasets.get("val")

    if isinstance(val_dataset, dict):
        val_dataset["dataroot_gt"] = VAL_GT
        val_dataset["dataroot_lq"] = VAL_LQ


def configure_student(
    config,
    stage1_g,
    stage1_le,
):
    config["name"] = (
        "train_HI_Diff_GoPro_"
        "Student_Lite_S2_Baseline"
    )

    config["num_gpu"] = 1
    config["manual_seed"] = 100

    network_g = config.get("network_g")

    if not isinstance(network_g, dict):
        raise KeyError(
            "設定檔缺少 network_g"
        )

    network_g["num_blocks"] = [
        1, 2, 2, 2
    ]
    network_g[
        "num_refinement_blocks"
    ] = 1

    network_g["dim"] = 48
    network_g["embed_dim"] = 64
    network_g["heads"] = [
        1, 2, 4, 8
    ]

    network_le_dm = config.get(
        "network_le_dm"
    )

    if not isinstance(
        network_le_dm, dict
    ):
        raise KeyError(
            "設定檔缺少 network_le_dm"
        )

    network_le_dm["embed_dim"] = 64
    network_le_dm["block_num"] = 6

    network_d = config.get("network_d")

    if not isinstance(network_d, dict):
        raise KeyError(
            "設定檔缺少 network_d"
        )

    network_d["in_channel"] = 256
    network_d["out_channel"] = 256
    network_d["inner_channel"] = 512
    network_d["block_num"] = 4

    path_config = config.setdefault(
        "path", {}
    )

    path_config[
        "pretrain_network_g"
    ] = str(stage1_g.resolve())

    path_config["param_key_g"] = "params"
    path_config["strict_load_g"] = True

    path_config[
        "pretrain_network_le"
    ] = str(stage1_le.resolve())

    path_config[
        "strict_load_le"
    ] = True

    path_config[
        "pretrain_network_le_dm"
    ] = None

    path_config[
        "strict_load_le_dm"
    ] = True

    path_config[
        "pretrain_network_d"
    ] = None

    path_config[
        "strict_load_d"
    ] = True

    path_config["resume_state"] = None

    configure_datasets(
        config,
        smoke=False,
    )


def configure_smoke(config):
    config["name"] = (
        "smoke_HI_Diff_GoPro_"
        "Student_Lite_S2_Baseline"
    )

    config["num_gpu"] = 1

    configure_datasets(
        config,
        smoke=True,
    )

    train_config = config.setdefault(
        "train", {}
    )

    train_config["total_iter"] = 20

    for key in (
        "mini_batch_sizes",
        "batch_sizes",
    ):
        value = train_config.get(key)

        if isinstance(value, list):
            train_config[key] = [
                1 for _ in value
            ]

    logger_config = config.setdefault(
        "logger", {}
    )

    logger_config["print_freq"] = 1
    logger_config[
        "save_checkpoint_freq"
    ] = 10

    val_config = config.setdefault(
        "val", {}
    )

    val_config["val_freq"] = 1000
    val_config["save_img"] = False


def save_yaml(config, path):
    with path.open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )


if not SOURCE_CONFIG.exists():
    raise FileNotFoundError(
        f"找不到官方 Stage 2 config："
        f"{SOURCE_CONFIG}"
    )

verify_directory(
    Path(TRAIN_GT),
    "GoPro train sharp",
)

verify_directory(
    Path(TRAIN_LQ),
    "GoPro train blur",
)

verify_directory(
    Path(VAL_GT),
    "GoPro test sharp",
)

verify_directory(
    Path(VAL_LQ),
    "GoPro test blur",
)

stage1_experiment = (
    find_stage1_experiment()
)

models_dir = (
    stage1_experiment / "models"
)

stage1_g = find_checkpoint(
    models_dir,
    "net_g",
)

stage1_le = find_checkpoint(
    models_dir,
    "net_le",
)

with SOURCE_CONFIG.open(
    "r", encoding="utf-8"
) as file:
    original = yaml.safe_load(file)

if not isinstance(original, dict):
    raise RuntimeError(
        f"無法解析：{SOURCE_CONFIG}"
    )

formal = deepcopy(original)

configure_student(
    formal,
    stage1_g,
    stage1_le,
)

smoke = deepcopy(formal)
configure_smoke(smoke)

save_yaml(formal, FORMAL_CONFIG)
save_yaml(smoke, SMOKE_CONFIG)

print()
print(
    "Student-Lite Stage 2 "
    "Baseline config 已建立"
)
print()
print(
    f"Stage 1 experiment: "
    f"{stage1_experiment}"
)
print(f"Stage 1 net_g: {stage1_g}")
print(f"Stage 1 net_le: {stage1_le}")
print()
print(
    f"Formal config: {FORMAL_CONFIG}"
)
print(
    f"Smoke config: {SMOKE_CONFIG}"
)
print()
print("Student-Lite architecture")
print("  num_blocks: [1, 2, 2, 2]")
print("  num_refinement_blocks: 1")
print("  dim: 48")
print("  embed_dim: 64")
print()
print("Expected inference parameters")
print("  Transformer: 11,524,730")
print("  LE-DM:          558,040")
print("  Denoising:    2,629,937")
print("  Total:       14,712,707")
print("  Reduction:       48.35%")
