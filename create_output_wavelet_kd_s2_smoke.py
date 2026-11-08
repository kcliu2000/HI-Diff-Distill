from pathlib import Path
import yaml


SOURCE = Path(
    "options/train/"
    "GoPro_Student_Lite_S2_OutputKD_smoke.yml"
)

TARGET = Path(
    "options/train/"
    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_smoke.yml"
)


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)


with SOURCE.open(
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)


config["name"] = (
    "Student-Lite-S2-"
    "OutputWaveletKD-Smoke"
)

config["model_type"] = (
    "HI_Diff_S2_OutputWaveletKD"
)


config["train"]["output_kd_opt"] = {
    "type": "L1Loss",
    "loss_weight": 0.5,
    "reduction": "mean",
}


config["train"]["wavelet_kd_opt"] = {
    "loss_weight": 0.5,
    "reduction": "mean",
    "ll_weight": 1.0,
    "lh_weight": 2.0,
    "hl_weight": 2.0,
    "hh_weight": 2.0,
}


config["train"]["total_iter"] = 20

config["datasets"]["train"][
    "batch_size_per_gpu"
] = 2

config["datasets"]["train"][
    "mini_batch_sizes"
] = [2]

config["datasets"]["train"][
    "iters"
] = [20]

config["datasets"]["train"][
    "gt_size"
] = 128

config["datasets"]["train"][
    "gt_sizes"
] = [128]


scheduler = config["train"].get(
    "scheduler"
)

if isinstance(scheduler, dict):
    scheduler["periods"] = [20]
    scheduler["restart_weights"] = [1]
    scheduler["eta_mins"] = [1e-6]


config["logger"]["print_freq"] = 1

config["logger"][
    "save_checkpoint_freq"
] = 10

config["logger"]["wandb"][
    "project"
] = "HI-Diff-Distill"

config["logger"]["wandb"][
    "resume_id"
] = None


config["val"]["val_freq"] = 1000
config["val"]["save_img"] = False


with TARGET.open(
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


print("Created:", TARGET)
print()
print("Model type:", config["model_type"])
print(
    "Output KD alpha:",
    config["train"][
        "output_kd_opt"
    ]["loss_weight"],
)
print(
    "Wavelet KD beta:",
    config["train"][
        "wavelet_kd_opt"
    ]["loss_weight"],
)
print(
    "Sub-band weights:",
    {
        key: config["train"][
            "wavelet_kd_opt"
        ][key]
        for key in (
            "ll_weight",
            "lh_weight",
            "hl_weight",
            "hh_weight",
        )
    },
)
