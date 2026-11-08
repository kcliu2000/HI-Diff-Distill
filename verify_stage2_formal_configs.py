from pathlib import Path
import yaml


TRAIN_DIR = Path("options/train")

CONFIGS = {
    "OutputKD-a050": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputKD_a050_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputKD"
        ),
        "alpha": 0.5,
        "beta": None,
        "hf": None,
    },
    "Main-a050-b050-hf200": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf200_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 2.0,
    },
    "a025-b025-hf200": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a025_b025_hf200_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.25,
        "beta": 0.25,
        "hf": 2.0,
    },
    "a075-b075-hf200": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a075_b075_hf200_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.75,
        "beta": 0.75,
        "hf": 2.0,
    },
    "a050-b050-hf150": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf150_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 1.5,
    },
    "a050-b050-hf250": {
        "path": (
            TRAIN_DIR
            / "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf250_Formal.yml"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 2.5,
    },
}

EXPECTED_G = (
    "/home/m11302124/HI-Diff/"
    "experiments/Student-Lite-S1-Formal/"
    "models/net_g_290000.pth"
)

EXPECTED_LE = (
    "/home/m11302124/HI-Diff/"
    "experiments/Student-Lite-S1-Formal/"
    "models/net_le_290000.pth"
)

EXPECTED_GT_SIZES = [
    128,
    160,
    192,
    256,
    320,
    384,
]

EXPECTED_BATCHES = [
    12,
    8,
    6,
    4,
    2,
    2,
]

EXPECTED_ITERS = [
    20000,
    20000,
    20000,
    20000,
    20000,
    200000,
]


for name, expected in CONFIGS.items():
    path = expected["path"]

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    train_opt = config["train"]
    dataset = config[
        "datasets"
    ]["train"]
    path_opt = config["path"]

    assert (
        config["model_type"]
        == expected["model_type"]
    )

    assert config["manual_seed"] == 100
    assert train_opt["total_iter"] == 300000
    assert dataset["gt_sizes"] == EXPECTED_GT_SIZES

    assert (
        dataset["mini_batch_sizes"]
        == EXPECTED_BATCHES
    )

    assert dataset["iters"] == EXPECTED_ITERS

    assert (
        train_opt["output_kd_opt"][
            "loss_weight"
        ]
        == expected["alpha"]
    )

    assert (
        path_opt["pretrain_network_g"]
        == EXPECTED_G
    )

    assert (
        path_opt["pretrain_network_le"]
        == EXPECTED_LE
    )

    assert (
        path_opt["resume_state"]
        is None
    )

    if expected["beta"] is None:
        assert (
            "wavelet_kd_opt"
            not in train_opt
        )
    else:
        wavelet = train_opt[
            "wavelet_kd_opt"
        ]

        assert (
            wavelet["loss_weight"]
            == expected["beta"]
        )

        assert wavelet["ll_weight"] == 1.0

        for key in (
            "lh_weight",
            "hl_weight",
            "hh_weight",
        ):
            assert (
                wavelet[key]
                == expected["hf"]
            )

    print(
        f"PASS: {name} | "
        f"alpha={expected['alpha']} | "
        f"beta={expected['beta']} | "
        f"HF={expected['hf']}"
    )


print()
print("All six formal Stage 2 configs: PASS")
