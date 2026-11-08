from copy import deepcopy
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"

SOURCES = {
    "OutputKD": (
        TRAIN_DIR
        / "GoPro_Student_Lite_S2_OutputKD_smoke.yml"
    ),
    "OutputWaveletKD": (
        TRAIN_DIR
        / "GoPro_Student_Lite_S2_OutputWaveletKD_smoke.yml"
    ),
}

BATCH_SIZES = (2, 1)
PATCH_SIZE = 384
TOTAL_ITER = 10


def make_config(
    source,
    method,
    batch_size,
):
    with source.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    config["name"] = (
        f"Bench-Student-Lite-S2-"
        f"{method}-P384-B{batch_size}"
    )

    config["num_gpu"] = 1
    config["manual_seed"] = 100

    train_dataset = config[
        "datasets"
    ]["train"]

    train_dataset[
        "batch_size_per_gpu"
    ] = batch_size

    train_dataset[
        "mini_batch_sizes"
    ] = [batch_size]

    train_dataset["gt_size"] = PATCH_SIZE
    train_dataset["gt_sizes"] = [PATCH_SIZE]
    train_dataset["iters"] = [TOTAL_ITER]
    train_dataset[
        "num_worker_per_gpu"
    ] = 2

    config["train"][
        "total_iter"
    ] = TOTAL_ITER

    scheduler = config[
        "train"
    ].get("scheduler")

    if isinstance(scheduler, dict):
        scheduler["periods"] = [
            TOTAL_ITER
        ]
        scheduler[
            "restart_weights"
        ] = [1]
        scheduler["eta_mins"] = [
            1e-6
        ]

    config["logger"][
        "print_freq"
    ] = 1

    config["logger"][
        "save_checkpoint_freq"
    ] = 1000

    # Benchmark 不需要建立 W&B run
    config["logger"]["wandb"][
        "project"
    ] = None

    config["logger"]["wandb"][
        "resume_id"
    ] = None

    config["val"]["val_freq"] = 1000
    config["val"]["save_img"] = False

    destination = (
        TRAIN_DIR
        / (
            "GoPro_Student_Lite_S2_"
            f"{method}_p384_b"
            f"{batch_size}_bench.yml"
        )
    )

    with destination.open(
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

    print("Created:", destination)


for method, source in SOURCES.items():
    if not source.exists():
        raise FileNotFoundError(source)

    for batch_size in BATCH_SIZES:
        make_config(
            source,
            method,
            batch_size,
        )
