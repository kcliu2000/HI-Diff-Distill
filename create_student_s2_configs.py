from copy import deepcopy
from pathlib import Path
import re
import sys
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"
EXPERIMENTS_DIR = ROOT / "experiments"

SOURCE_CANDIDATES = [
    TRAIN_DIR / "GoPro_S2.yml",
    TRAIN_DIR / "GoPro_S2.yaml",
    TRAIN_DIR / "GoPro_stage2.yml",
    TRAIN_DIR / "GoPro_stage2.yaml",
]


def numbered_suffix(path):
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def find_source_config():
    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = sorted(
        list(TRAIN_DIR.glob("*GoPro*S2*.yml"))
        + list(TRAIN_DIR.glob("*GoPro*S2*.yaml"))
        + list(TRAIN_DIR.glob("*GoPro*stage2*.yml"))
        + list(TRAIN_DIR.glob("*GoPro*stage2*.yaml"))
    )

    if not candidates:
        print("找不到 GoPro Stage 2 設定檔。")
        sys.exit(1)

    return candidates[0]


def find_stage1_checkpoint():
    search_roots = [
        EXPERIMENTS_DIR /
        "smoke_HI_Diff_GoPro_Student_S1" /
        "models",
        EXPERIMENTS_DIR /
        "resume_HI_Diff_GoPro_Student_S1" /
        "models",
    ]

    candidates = []

    for directory in search_roots:
        if not directory.exists():
            continue

        candidates.extend(directory.glob("net_g_*.pth"))

    candidates = [
        path for path in candidates
        if "latest" not in path.name.lower()
    ]

    if not candidates:
        latest_candidates = []

        for directory in search_roots:
            if directory.exists():
                latest_candidates.extend(
                    directory.glob("*latest*.pth")
                )

        if latest_candidates:
            return sorted(latest_candidates)[-1]

        raise FileNotFoundError(
            "找不到 Student Stage 1 net_g checkpoint。"
        )

    return max(candidates, key=numbered_suffix)


def find_stage1_latent_checkpoint(net_g_path):
    directory = net_g_path.parent
    iteration = numbered_suffix(net_g_path)

    patterns = [
        f"net_le_{iteration}.pth",
        f"net_le_latest.pth",
        f"*le*_{iteration}.pth",
        "*le*latest*.pth",
    ]

    for pattern in patterns:
        matches = sorted(directory.glob(pattern))

        if matches:
            return matches[-1]

    raise FileNotFoundError(
        "找不到與 Student Stage 1 對應的 "
        "latent encoder checkpoint。"
    )


def configure_dataset(dataset_opt, smoke=False):
    if not isinstance(dataset_opt, dict):
        return

    if "num_worker_per_gpu" in dataset_opt:
        dataset_opt["num_worker_per_gpu"] = 2

    if "batch_size_per_gpu" in dataset_opt:
        dataset_opt["batch_size_per_gpu"] = 1

    if smoke:
        for key in ("mini_batch_sizes", "batch_sizes"):
            if (
                key in dataset_opt
                and isinstance(dataset_opt[key], list)
            ):
                dataset_opt[key] = (
                    [1] * len(dataset_opt[key])
                )


def configure_student(
    config,
    stage1_g,
    stage1_le,
):
    config["name"] = (
        "train_HI_Diff_GoPro_Student_S2_Baseline"
    )
    config["num_gpu"] = 1
    config["manual_seed"] = 100

    network_g = config["network_g"]
    network_g["num_blocks"] = [2, 3, 4, 4]
    network_g["num_refinement_blocks"] = 2

    path_opt = config.setdefault("path", {})

    path_opt["pretrain_network_g"] = str(
        stage1_g.resolve()
    )
    path_opt["strict_load_g"] = True

    stage1_keys = [
        "pretrain_network_le",
        "pretrain_network_le_s1",
    ]

    latent_key_set = False

    for key in stage1_keys:
        if key in path_opt:
            path_opt[key] = str(stage1_le.resolve())
            latent_key_set = True

    if not latent_key_set:
        path_opt["pretrain_network_le"] = str(
            stage1_le.resolve()
        )

    for key in (
        "strict_load_le",
        "strict_load_le_s1",
    ):
        if key in path_opt:
            path_opt[key] = True

    path_opt["resume_state"] = None

    datasets = config.get("datasets", {})

    if isinstance(datasets, dict):
        for dataset_opt in datasets.values():
            configure_dataset(dataset_opt, smoke=False)


def configure_smoke(config):
    config["name"] = (
        "smoke_HI_Diff_GoPro_Student_S2_Baseline"
    )

    train_opt = config.setdefault("train", {})
    train_opt["total_iter"] = 20

    for key in ("mini_batch_sizes", "batch_sizes"):
        if (
            key in train_opt
            and isinstance(train_opt[key], list)
        ):
            train_opt[key] = (
                [1] * len(train_opt[key])
            )

    logger_opt = config.setdefault("logger", {})
    logger_opt["print_freq"] = 1
    logger_opt["save_checkpoint_freq"] = 10

    val_opt = config.get("val")

    if isinstance(val_opt, dict):
        val_opt["val_freq"] = 1000

    datasets = config.get("datasets", {})

    if isinstance(datasets, dict):
        for phase, dataset_opt in datasets.items():
            configure_dataset(
                dataset_opt,
                smoke=(str(phase).lower() == "train"),
            )


def save_yaml(config, path):
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )


source = find_source_config()
stage1_g = find_stage1_checkpoint()
stage1_le = find_stage1_latent_checkpoint(stage1_g)

with source.open("r", encoding="utf-8") as file:
    original = yaml.safe_load(file)

student = deepcopy(original)

configure_student(
    student,
    stage1_g,
    stage1_le,
)

smoke = deepcopy(student)
configure_smoke(smoke)

student_path = (
    TRAIN_DIR /
    "GoPro_Student_S2_Baseline.yml"
)

smoke_path = (
    TRAIN_DIR /
    "GoPro_Student_S2_Baseline_smoke.yml"
)

save_yaml(student, student_path)
save_yaml(smoke, smoke_path)

print("已建立 Student Stage 2 baseline 設定")
print(f"官方來源: {source}")
print(f"Stage 1 net_g: {stage1_g}")
print(f"Stage 1 net_le: {stage1_le}")
print(f"正式設定: {student_path}")
print(f"Smoke 設定: {smoke_path}")
