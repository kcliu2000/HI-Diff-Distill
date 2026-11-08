from copy import deepcopy
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "options/test/GoPro_only.yml"

OUTPUT = (
    ROOT
    / "options/test/GoPro_train_teacher_outputs.yml"
)

TRAIN_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/sharp"
)

TRAIN_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/blur"
)


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)


with SOURCE.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


config["name"] = "generate_HI_Diff_GoPro_Train_Teacher"
config["model_type"] = "HI_Diff_S2"
config["num_gpu"] = 1
config["manual_seed"] = 100


config["datasets"] = {
    "test_1": {
        "name": "GoPro_Train_Teacher",
        "type": "PairedImageIRDataset",
        "dataroot_gt": TRAIN_GT,
        "dataroot_lq": TRAIN_LQ,
        "filename_tmpl": "{}",
        "io_backend": {
            "type": "disk",
        },
    }
}


config["val"]["save_img"] = True
config["val"]["suffix"] = None
config["val"]["selfensemble_testing"] = False


with OUTPUT.open("w", encoding="utf-8") as file:
    yaml.safe_dump(
        config,
        file,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )


print("Teacher-output config created")
print(f"Source: {SOURCE}")
print(f"Output: {OUTPUT}")
print(f"Train GT: {TRAIN_GT}")
print(f"Train LQ: {TRAIN_LQ}")

print()
print("Teacher checkpoints:")

for key in (
    "pretrain_network_g",
    "pretrain_network_le_dm",
    "pretrain_network_d",
):
    print(
        f"  {key}:",
        config["path"].get(key),
    )
