from copy import deepcopy
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "options/train/GoPro_S1.yml"
TRAIN_DIR = ROOT / "options/train"

FORMAL_OUTPUT = (
    TRAIN_DIR /
    "GoPro_Student50_S1.yml"
)

SMOKE_OUTPUT = (
    TRAIN_DIR /
    "GoPro_Student50_S1_smoke.yml"
)


def configure_dataset(dataset, smoke=False):
    if not isinstance(dataset, dict):
        return

    if "num_worker_per_gpu" in dataset:
        dataset["num_worker_per_gpu"] = 2

    if "batch_size_per_gpu" in dataset:
        dataset["batch_size_per_gpu"] = 1

    if smoke:
        for key in ("mini_batch_sizes", "batch_sizes"):
            value = dataset.get(key)

            if isinstance(value, list):
                dataset[key] = [1] * len(value)


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


with SOURCE.open("r", encoding="utf-8") as file:
    original = yaml.safe_load(file)

formal = deepcopy(original)

formal["name"] = "train_HI_Diff_GoPro_Student50_S1"
formal["num_gpu"] = 1
formal["manual_seed"] = 100

formal["network_g"]["num_blocks"] = [1, 2, 2, 2]
formal["network_g"]["num_refinement_blocks"] = 1

path_opt = formal.setdefault("path", {})
path_opt["pretrain_network_g"] = None
path_opt["strict_load_g"] = False
path_opt["pretrain_network_le"] = None
path_opt["strict_load_le"] = False
path_opt["resume_state"] = None

for dataset in formal.get("datasets", {}).values():
    configure_dataset(dataset, smoke=False)

smoke = deepcopy(formal)

smoke["name"] = "smoke_HI_Diff_GoPro_Student50_S1"
smoke["train"]["total_iter"] = 20
smoke["logger"]["print_freq"] = 1
smoke["logger"]["save_checkpoint_freq"] = 10
smoke["val"]["val_freq"] = 1000
smoke["val"]["save_img"] = False

for phase, dataset in smoke.get("datasets", {}).items():
    configure_dataset(
        dataset,
        smoke=(str(phase).lower() == "train"),
    )

save_yaml(formal, FORMAL_OUTPUT)
save_yaml(smoke, SMOKE_OUTPUT)

print(f"Created: {FORMAL_OUTPUT}")
print(f"Created: {SMOKE_OUTPUT}")
print()
print("Student50 architecture:")
print(
    "  num_blocks:",
    formal["network_g"]["num_blocks"],
)
print(
    "  num_refinement_blocks:",
    formal["network_g"]["num_refinement_blocks"],
)
print("  dim:", formal["network_g"]["dim"])
print("  embed_dim:", formal["network_g"]["embed_dim"])
