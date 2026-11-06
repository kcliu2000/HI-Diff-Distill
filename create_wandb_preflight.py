from pathlib import Path
import yaml


SOURCE = Path(
    "options/train/"
    "GoPro_Student_Lite_S1_Formal.yml"
)

TARGET = Path(
    "options/train/"
    "GoPro_Student_Lite_S1_WandB_smoke.yml"
)


with SOURCE.open(
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)


config["name"] = (
    "Student-Lite-S1-WandB-Smoke"
)

config["train"]["total_iter"] = 20

train_dataset = config["datasets"]["train"]

train_dataset["gt_size"] = 128
train_dataset["gt_sizes"] = [128]
train_dataset["batch_size_per_gpu"] = 2
train_dataset["mini_batch_sizes"] = [2]
train_dataset["iters"] = [20]
train_dataset["num_worker_per_gpu"] = 2


scheduler = config["train"].get(
    "scheduler"
)

if isinstance(scheduler, dict):
    scheduler["periods"] = [20]
    scheduler["restart_weights"] = [1]
    scheduler["eta_mins"] = [1e-6]


config["logger"]["print_freq"] = 1
config["logger"]["save_checkpoint_freq"] = 10

config["logger"]["wandb"]["project"] = (
    "HI-Diff-Distill"
)

config["logger"]["wandb"]["resume_id"] = None

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


print(f"Created: {TARGET}")
print(
    "W&B project:",
    config["logger"]["wandb"]["project"],
)
print("Run name:", config["name"])
