from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"
TEST_DIR = ROOT / "options" / "test"

TEST_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/sharp"
)

TEST_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/blur"
)

STAGE1_LE = (
    ROOT
    / "experiments"
    / "Student-Lite-S1-Formal"
    / "models"
    / "net_le_290000.pth"
)

EXPERIMENTS = [
    {
        "name": "Student-Lite-S2-OutputKD-a050-Best-Test",
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputKD_a050_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputKD_a050_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputkd_a050_best"
        ),
    },
    {
        "name": (
            "Student-Lite-S2-OutputWaveletKD-"
            "a025-b025-hf200-Best-Test"
        ),
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a025_b025_hf200_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a025_b025_hf200_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputwaveletkd_"
            "a025_b025_hf200_best"
        ),
    },
    {
        "name": (
            "Student-Lite-S2-OutputWaveletKD-"
            "a050-b050-hf150-Best-Test"
        ),
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf150_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf150_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputwaveletkd_"
            "a050_b050_hf150_best"
        ),
    },
    {
        "name": (
            "Student-Lite-S2-OutputWaveletKD-"
            "a050-b050-hf200-Best-Test"
        ),
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf200_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf200_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputwaveletkd_"
            "a050_b050_hf200_best"
        ),
    },
    {
        "name": (
            "Student-Lite-S2-OutputWaveletKD-"
            "a050-b050-hf250-Best-Test"
        ),
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf250_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf250_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputwaveletkd_"
            "a050_b050_hf250_best"
        ),
    },
    {
        "name": (
            "Student-Lite-S2-OutputWaveletKD-"
            "a075-b075-hf200-Best-Test"
        ),
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a075_b075_hf200_Best.yml"
        ),
        "train_config": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a075_b075_hf200_Formal.yml"
        ),
        "best_dir": (
            "student_lite_s2_outputwaveletkd_"
            "a075_b075_hf200_best"
        ),
    },
]


def load_yaml(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise RuntimeError(
            f"Invalid YAML: {path}"
        )

    return config


if not STAGE1_LE.exists():
    raise FileNotFoundError(STAGE1_LE)


for experiment in EXPERIMENTS:
    train_path = (
        TRAIN_DIR
        / experiment["train_config"]
    )

    train_config = load_yaml(train_path)

    best_dir = (
        ROOT
        / "experiments"
        / experiment["best_dir"]
    )

    net_g = best_dir / "net_g.pth"
    net_le_dm = best_dir / "net_le_dm.pth"
    net_d = best_dir / "net_d.pth"

    for checkpoint in (
        net_g,
        net_le_dm,
        net_d,
    ):
        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

    config = {
        "name": experiment["name"],
        "model_type": "HI_Diff_S2",
        "scale": 1,
        "num_gpu": 1,
        "manual_seed": 100,
        "datasets": {
            "test_1": {
                "name": "GoPro",
                "type": "PairedImageIRDataset",
                "dataroot_gt": TEST_GT,
                "dataroot_lq": TEST_LQ,
                "filename_tmpl": "{}",
                "io_backend": {
                    "type": "disk",
                },
            }
        },
        "network_g": deepcopy(
            train_config["network_g"]
        ),
        "network_le": deepcopy(
            train_config["network_le"]
        ),
        "network_le_dm": deepcopy(
            train_config["network_le_dm"]
        ),
        "network_d": deepcopy(
            train_config["network_d"]
        ),
        "diffusion_schedule": deepcopy(
            train_config[
                "diffusion_schedule"
            ]
        ),
        "path": {
            "pretrain_network_g": str(
                net_g.resolve()
            ),
            "param_key_g": "params",
            "strict_load_g": True,
            "pretrain_network_le": str(
                STAGE1_LE.resolve()
            ),
            "param_key_le": "params",
            "strict_load_le": True,
            "pretrain_network_le_dm": str(
                net_le_dm.resolve()
            ),
            "param_key_le_dm": "params",
            "strict_load_le_dm": True,
            "pretrain_network_d": str(
                net_d.resolve()
            ),
            "param_key_d": "params",
            "strict_load_d": True,
            "resume_state": None,
        },
        "val": {
            "save_img": False,
            "suffix": "",
            "selfensemble_testing": False,
            "pbar": True,
            "metrics": {
                "psnr": {
                    "type": "calculate_psnr",
                    "crop_border": 0,
                    "test_y_channel": False,
                },
                "ssim": {
                    "type": "calculate_ssim",
                    "crop_border": 0,
                    "test_y_channel": False,
                },
            },
        },
    }

    destination = (
        TEST_DIR
        / experiment["filename"]
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    print("Created:", destination)
    print("  net_g:", net_g.resolve())
    print(
        "  net_le_dm:",
        net_le_dm.resolve(),
    )
    print("  net_d:", net_d.resolve())


print()
print("All six Stage 2 test configs created.")
