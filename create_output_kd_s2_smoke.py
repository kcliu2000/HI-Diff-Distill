from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT
    / "options/train/"
    "GoPro_Student_Lite_S2_Baseline.yml"
)

OUTPUT = (
    ROOT
    / "options/train/"
    "GoPro_Student_Lite_S2_OutputKD_smoke.yml"
)

TRAIN_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/sharp"
)

TRAIN_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/blur"
)

TRAIN_TEACHER = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/train/teacher_hidiff"
)

VAL_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/sharp"
)

VAL_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/blur"
)

STAGE1_G = (
    ROOT
    / "experiments/student_lite_stage1_best/"
    "net_g.pth"
)

STAGE1_LE = (
    ROOT
    / "experiments/student_lite_stage1_best/"
    "net_le.pth"
)


for path in (
    SOURCE,
    Path(TRAIN_GT),
    Path(TRAIN_LQ),
    Path(TRAIN_TEACHER),
    Path(VAL_GT),
    Path(VAL_LQ),
    STAGE1_G,
    STAGE1_LE,
):
    if not path.exists():
        raise FileNotFoundError(path)


with SOURCE.open(
    "r",
    encoding="utf-8",
) as file:
    config = deepcopy(
        yaml.safe_load(file)
    )


config["name"] = (
    "Student-Lite-S2-OutputKD-Smoke"
)

config["model_type"] = (
    "HI_Diff_S2_OutputKD"
)

config["num_gpu"] = 1
config["manual_seed"] = 100


train_set = config["datasets"]["train"]

train_set["name"] = "GoProTrainOutputKD"
train_set["type"] = (
    "PairedImageTeacherDataset"
)

train_set["dataroot_gt"] = TRAIN_GT
train_set["dataroot_lq"] = TRAIN_LQ
train_set[
    "dataroot_teacher"
] = TRAIN_TEACHER

train_set["num_worker_per_gpu"] = 2
train_set["batch_size_per_gpu"] = 2

train_set["gt_size"] = 128
train_set["gt_sizes"] = [128]
train_set["mini_batch_sizes"] = [2]
train_set["iters"] = [20]

train_set["use_hflip"] = True
train_set["use_rot"] = True


val_set = config["datasets"]["val"]

val_set["name"] = "GoProVal"
val_set["type"] = "PairedImageIRDataset"
val_set["dataroot_gt"] = VAL_GT
val_set["dataroot_lq"] = VAL_LQ

val_set.pop(
    "dataroot_teacher",
    None,
)


config["network_g"]["num_blocks"] = [
    1,
    2,
    2,
    2,
]

config["network_g"][
    "num_refinement_blocks"
] = 1


path_opt = config["path"]

path_opt["pretrain_network_g"] = str(
    STAGE1_G.resolve()
)

path_opt["param_key_g"] = "params"
path_opt["strict_load_g"] = True

path_opt["pretrain_network_le"] = str(
    STAGE1_LE.resolve()
)

path_opt["param_key_le"] = "params"
path_opt["strict_load_le"] = True

path_opt["pretrain_network_le_dm"] = None
path_opt["pretrain_network_d"] = None
path_opt["resume_state"] = None


train_opt = config["train"]

train_opt["total_iter"] = 20

train_opt["output_kd_opt"] = {
    "type": "L1Loss",
    "loss_weight": 0.5,
    "reduction": "mean",
}


scheduler = train_opt.get("scheduler")

if isinstance(scheduler, dict):
    scheduler["periods"] = [20]
    scheduler["restart_weights"] = [1]
    scheduler["eta_mins"] = [1e-6]


logger = config["logger"]

logger["print_freq"] = 1
logger["save_checkpoint_freq"] = 10
logger["use_tb_logger"] = True

logger["wandb"] = {
    "project": "HI-Diff-Distill",
    "resume_id": None,
}


config["val"]["val_freq"] = 1000
config["val"]["save_img"] = False


with OUTPUT.open(
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


print("Created:", OUTPUT)
print("Model:", config["model_type"])
print(
    "Stage 1 net_g:",
    path_opt["pretrain_network_g"],
)
print(
    "Stage 1 net_le:",
    path_opt["pretrain_network_le"],
)
print(
    "Output KD alpha:",
    train_opt[
        "output_kd_opt"
    ]["loss_weight"],
)
