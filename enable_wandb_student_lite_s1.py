from pathlib import Path
import shutil
import yaml


CONFIG_PATH = Path(
    "options/train/GoPro_Student_Lite_S1_Formal.yml"
)

PROJECT_NAME = "HI-Diff-Distill"
RUN_NAME = "Student-Lite-S1-Formal"


if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config not found: {CONFIG_PATH}"
    )

backup_path = CONFIG_PATH.with_suffix(
    CONFIG_PATH.suffix + ".before_wandb"
)

if not backup_path.exists():
    shutil.copy2(
        CONFIG_PATH,
        backup_path,
    )
    print(f"Backup created: {backup_path}")
else:
    print(f"Backup already exists: {backup_path}")


with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    config = yaml.safe_load(file)


if not isinstance(config, dict):
    raise RuntimeError(
        f"Cannot parse YAML: {CONFIG_PATH}"
    )


config["name"] = RUN_NAME

logger = config.setdefault(
    "logger",
    {},
)

logger["print_freq"] = 100
logger["save_checkpoint_freq"] = 10000
logger["use_tb_logger"] = True

wandb_config = logger.setdefault(
    "wandb",
    {},
)

wandb_config["project"] = PROJECT_NAME
wandb_config["resume_id"] = None


with CONFIG_PATH.open(
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


print()
print("W&B enabled")
print(f"Config: {CONFIG_PATH}")
print(f"Project: {PROJECT_NAME}")
print(f"Run name: {RUN_NAME}")
print(
    "TensorBoard logging:",
    logger["use_tb_logger"],
)
print(
    "W&B resume ID:",
    wandb_config["resume_id"],
)
