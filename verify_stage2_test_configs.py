from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "options" / "test"

CONFIGS = [
    "GoPro_Student_Lite_S2_"
    "OutputKD_a050_Best.yml",

    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_"
    "a025_b025_hf200_Best.yml",

    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_"
    "a050_b050_hf150_Best.yml",

    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_"
    "a050_b050_hf200_Best.yml",

    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_"
    "a050_b050_hf250_Best.yml",

    "GoPro_Student_Lite_S2_"
    "OutputWaveletKD_"
    "a075_b075_hf200_Best.yml",
]

EXPECTED_GT = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/sharp"
)

EXPECTED_LQ = (
    "/home/m11302124/MIMO-UNet-Wavelet/"
    "dataset/GOPRO/test/blur"
)

EXPECTED_LE = (
    "/home/m11302124/HI-Diff/"
    "experiments/Student-Lite-S1-Formal/"
    "models/net_le_290000.pth"
)


for filename in CONFIGS:
    path = TEST_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assert (
        config["model_type"]
        == "HI_Diff_S2"
    )

    assert config["manual_seed"] == 100
    assert config["num_gpu"] == 1

    dataset = config[
        "datasets"
    ]["test_1"]

    assert (
        dataset["dataroot_gt"]
        == EXPECTED_GT
    )

    assert (
        dataset["dataroot_lq"]
        == EXPECTED_LQ
    )

    network_g = config["network_g"]

    assert (
        network_g["num_blocks"]
        == [1, 2, 2, 2]
    )

    assert (
        network_g[
            "num_refinement_blocks"
        ]
        == 1
    )

    assert network_g["dim"] == 48
    assert network_g["embed_dim"] == 64
    assert network_g["group"] == 4

    path_opt = config["path"]

    assert (
        path_opt[
            "pretrain_network_le"
        ]
        == EXPECTED_LE
    )

    for key in (
        "pretrain_network_g",
        "pretrain_network_le",
        "pretrain_network_le_dm",
        "pretrain_network_d",
    ):
        checkpoint = Path(
            path_opt[key]
        )

        assert checkpoint.exists(), (
            f"Missing checkpoint: "
            f"{checkpoint}"
        )

    assert (
        path_opt["param_key_g"]
        == "params"
    )

    assert (
        path_opt["param_key_le"]
        == "params"
    )

    assert (
        path_opt["param_key_le_dm"]
        == "params"
    )

    assert (
        path_opt["param_key_d"]
        == "params"
    )

    metrics = config["val"]["metrics"]

    assert set(metrics) == {
        "psnr",
        "ssim",
    }

    for metric in (
        "psnr",
        "ssim",
    ):
        assert (
            metrics[metric][
                "crop_border"
            ]
            == 0
        )

        assert (
            metrics[metric][
                "test_y_channel"
            ]
            is False
        )

    assert (
        config["val"]["save_img"]
        is False
    )

    print(
        "PASS:",
        path.name,
    )


print()
print("All six Stage 2 test configs: PASS")
