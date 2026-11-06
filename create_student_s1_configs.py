from pathlib import Path
from copy import deepcopy
import sys
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "options" / "train"

SOURCE_CANDIDATES = [
    TRAIN_DIR / "GoPro_S1.yml",
    TRAIN_DIR / "GoPro_S1.yaml",
    TRAIN_DIR / "GoPro_stage1.yml",
    TRAIN_DIR / "GoPro_stage1.yaml",
]


def find_source_config():
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path

    candidates = sorted(
        list(TRAIN_DIR.glob("*GoPro*S1*.yml"))
        + list(TRAIN_DIR.glob("*GoPro*S1*.yaml"))
        + list(TRAIN_DIR.glob("*GoPro*stage1*.yml"))
        + list(TRAIN_DIR.glob("*GoPro*stage1*.yaml"))
    )

    if candidates:
        return candidates[0]

    print("找不到 GoPro Stage 1 設定檔。")
    print("options/train 目前包含：")

    for path in sorted(TRAIN_DIR.glob("*")):
        if path.is_file():
            print(f"  {path.name}")

    sys.exit(1)


def set_if_present(mapping, key, value):
    if isinstance(mapping, dict) and key in mapping:
        mapping[key] = value
        return True
    return False


def configure_dataset(dataset_opt, smoke=False):
    if not isinstance(dataset_opt, dict):
        return

    set_if_present(dataset_opt, "num_worker_per_gpu", 2)
    set_if_present(dataset_opt, "batch_size_per_gpu", 1)

    if smoke:
        for key in (
            "mini_batch_sizes",
            "mini_batch_size",
            "batch_sizes",
        ):
            if key in dataset_opt:
                current = dataset_opt[key]

                if isinstance(current, list):
                    dataset_opt[key] = [1] * len(current)
                else:
                    dataset_opt[key] = 1


def configure_paths(config):
    path_opt = config.setdefault("path", {})

    path_opt["pretrain_network_g"] = None
    path_opt["strict_load_g"] = False
    path_opt["resume_state"] = None

    for key in (
        "pretrain_network_le",
        "pretrain_network_le_dm",
        "pretrain_network_d",
    ):
        if key in path_opt:
            path_opt[key] = None

    for key in (
        "strict_load_le",
        "strict_load_le_dm",
        "strict_load_d",
    ):
        if key in path_opt:
            path_opt[key] = False


def configure_student(config):
    config["name"] = "train_HI_Diff_GoPro_Student_S1"
    config["num_gpu"] = 1
    config["manual_seed"] = 100

    network_g = config.get("network_g")

    if not isinstance(network_g, dict):
        raise KeyError("設定檔中找不到 network_g。")

    network_g["num_blocks"] = [2, 3, 4, 4]
    network_g["num_refinement_blocks"] = 2

    expected_unchanged = {
        "dim": 48,
        "embed_dim": 64,
        "heads": [1, 2, 4, 8],
    }

    print("\nStudent network_g 核心設定：")

    for key, default_value in expected_unchanged.items():
        if key in network_g:
            print(f"  {key}: {network_g[key]}")
        else:
            print(
                f"  警告：network_g 沒有 {key}，"
                f"預期參考值為 {default_value}"
            )

    print(f"  num_blocks: {network_g['num_blocks']}")
    print(
        "  num_refinement_blocks: "
        f"{network_g['num_refinement_blocks']}"
    )

    datasets = config.get("datasets", {})

    if isinstance(datasets, dict):
        for dataset_opt in datasets.values():
            configure_dataset(dataset_opt, smoke=False)

    configure_paths(config)


def configure_smoke(config):
    config["name"] = "smoke_HI_Diff_GoPro_Student_S1"
    config["num_gpu"] = 1

    train_opt = config.setdefault("train", {})
    train_opt["total_iter"] = 20

    logger_opt = config.setdefault("logger", {})
    logger_opt["print_freq"] = 1
    logger_opt["save_checkpoint_freq"] = 10

    val_opt = config.get("val")

    if isinstance(val_opt, dict):
        val_opt["val_freq"] = 10

    datasets = config.get("datasets", {})

    if isinstance(datasets, dict):
        for phase, dataset_opt in datasets.items():
            configure_dataset(
                dataset_opt,
                smoke=(str(phase).lower() == "train"),
            )


def save_yaml(config, destination):
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )


source = find_source_config()

with source.open("r", encoding="utf-8") as file:
    original = yaml.safe_load(file)

if not isinstance(original, dict):
    raise RuntimeError(f"無法解析設定檔：{source}")

student = deepcopy(original)
configure_student(student)

smoke = deepcopy(student)
configure_smoke(smoke)

student_path = TRAIN_DIR / "GoPro_Student_S1.yml"
smoke_path = TRAIN_DIR / "GoPro_Student_S1_smoke.yml"

save_yaml(student, student_path)
save_yaml(smoke, smoke_path)

print("\n已建立設定檔：")
print(f"  官方來源：{source}")
print(f"  正式訓練：{student_path}")
print(f"  Smoke test：{smoke_path}")

print("\nSmoke test 設定：")
print("  num_gpu: 1")
print("  total_iter: 20")
print("  batch_size_per_gpu: 1")
print("  num_worker_per_gpu: 2")
print("  print_freq: 1")
print("  save_checkpoint_freq: 10")
