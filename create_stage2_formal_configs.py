from copy import deepcopy
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "options" / "train"

OUTPUTKD_SOURCE = (
    TRAIN_DIR
    / "GoPro_Student_Lite_S2_OutputKD_smoke.yml"
)

WAVELET_SOURCE = (
    TRAIN_DIR
    / "GoPro_Student_Lite_S2_OutputWaveletKD_smoke.yml"
)

STAGE1_G = (
    ROOT
    / "experiments"
    / "Student-Lite-S1-Formal"
    / "models"
    / "net_g_290000.pth"
)

STAGE1_LE = (
    ROOT
    / "experiments"
    / "Student-Lite-S1-Formal"
    / "models"
    / "net_le_290000.pth"
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

GT_SIZES = [
    128,
    160,
    192,
    256,
    320,
    384,
]

MINI_BATCH_SIZES = [
    12,
    8,
    6,
    4,
    2,
    2,
]

STAGE_ITERS = [
    20000,
    20000,
    20000,
    20000,
    20000,
    200000,
]

TOTAL_ITER = sum(STAGE_ITERS)

EXPERIMENTS = [
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputKD_a050_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputKD-a050-Formal"
        ),
        "model_type": "HI_Diff_S2_OutputKD",
        "alpha": 0.5,
        "beta": None,
        "hf": None,
    },
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf200_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputWaveletKD-"
            "a050-b050-hf200-Formal"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 2.0,
    },
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a025_b025_hf200_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputWaveletKD-"
            "a025-b025-hf200-Formal"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.25,
        "beta": 0.25,
        "hf": 2.0,
    },
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a075_b075_hf200_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputWaveletKD-"
            "a075-b075-hf200-Formal"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.75,
        "beta": 0.75,
        "hf": 2.0,
    },
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf150_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputWaveletKD-"
            "a050-b050-hf150-Formal"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 1.5,
    },
    {
        "filename": (
            "GoPro_Student_Lite_S2_"
            "OutputWaveletKD_"
            "a050_b050_hf250_Formal.yml"
        ),
        "name": (
            "Student-Lite-S2-"
            "OutputWaveletKD-"
            "a050-b050-hf250-Formal"
        ),
        "model_type": (
            "HI_Diff_S2_OutputWaveletKD"
        ),
        "alpha": 0.5,
        "beta": 0.5,
        "hf": 2.5,
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
            f"Cannot parse YAML: {path}"
        )

    return config


def configure_common(config, experiment):
    config["name"] = experiment["name"]
    config["model_type"] = experiment[
        "model_type"
    ]
    config["num_gpu"] = 1
    config["manual_seed"] = 100

    train_dataset = config[
        "datasets"
    ]["train"]

    train_dataset["name"] = (
        "GoProTrainKDFixedTeacher"
    )

    train_dataset["type"] = (
        "PairedImageTeacherDataset"
    )

    train_dataset["dataroot_gt"] = TRAIN_GT
    train_dataset["dataroot_lq"] = TRAIN_LQ

    train_dataset[
        "dataroot_teacher"
    ] = TRAIN_TEACHER

    train_dataset["use_shuffle"] = True

    train_dataset[
        "num_worker_per_gpu"
    ] = 8

    # DataLoader needs to provide the largest
    # progressive mini-batch.
    train_dataset[
        "batch_size_per_gpu"
    ] = max(MINI_BATCH_SIZES)

    train_dataset[
        "dataset_enlarge_ratio"
    ] = 1

    train_dataset["gt_size"] = max(GT_SIZES)
    train_dataset["gt_sizes"] = GT_SIZES

    train_dataset[
        "mini_batch_sizes"
    ] = MINI_BATCH_SIZES

    train_dataset["iters"] = STAGE_ITERS

    train_dataset["use_hflip"] = True
    train_dataset["use_rot"] = True

    val_dataset = config[
        "datasets"
    ]["val"]

    val_dataset["name"] = "GoProVal"
    val_dataset["dataroot_gt"] = VAL_GT
    val_dataset["dataroot_lq"] = VAL_LQ

    path_opt = config.setdefault(
        "path",
        {},
    )

    path_opt[
        "pretrain_network_g"
    ] = str(STAGE1_G.resolve())

    path_opt[
        "pretrain_network_le"
    ] = str(STAGE1_LE.resolve())

    path_opt["param_key_g"] = "params"
    path_opt["param_key_le"] = "params"

    path_opt["strict_load_g"] = True
    path_opt["strict_load_le"] = True

    path_opt[
        "pretrain_network_le_dm"
    ] = None

    path_opt[
        "pretrain_network_d"
    ] = None

    path_opt["resume_state"] = None

    train_opt = config["train"]
    train_opt["total_iter"] = TOTAL_ITER
    train_opt["warmup_iter"] = -1
    train_opt["use_grad_clip"] = True

    train_opt["output_kd_opt"] = {
        "type": "L1Loss",
        "loss_weight": experiment["alpha"],
        "reduction": "mean",
    }

    if experiment["beta"] is None:
        train_opt.pop(
            "wavelet_kd_opt",
            None,
        )
    else:
        train_opt["wavelet_kd_opt"] = {
            "loss_weight": experiment["beta"],
            "reduction": "mean",
            "ll_weight": 1.0,
            "lh_weight": experiment["hf"],
            "hl_weight": experiment["hf"],
            "hh_weight": experiment["hf"],
        }

    scheduler = train_opt.get(
        "scheduler"
    )

    if isinstance(scheduler, dict):
        scheduler["periods"] = [
            100000,
            200000,
        ]

        scheduler[
            "restart_weights"
        ] = [1, 1]

        scheduler["eta_mins"] = [
            0.0002,
            0.000001,
        ]

    logger = config.setdefault(
        "logger",
        {},
    )

    logger["print_freq"] = 100

    logger[
        "save_checkpoint_freq"
    ] = 10000

    logger["use_tb_logger"] = True

    logger["wandb"] = {
        "project": "HI-Diff-Distill",
        "resume_id": None,
    }

    val_opt = config.setdefault(
        "val",
        {},
    )

    val_opt["val_freq"] = 10000
    val_opt["save_img"] = False


def verify_config(config, experiment):
    train_dataset = config[
        "datasets"
    ]["train"]

    train_opt = config["train"]
    path_opt = config["path"]

    checks = [
        config["model_type"]
        == experiment["model_type"],

        config["manual_seed"] == 100,

        train_opt["total_iter"]
        == 300000,

        train_dataset["gt_sizes"]
        == GT_SIZES,

        train_dataset[
            "mini_batch_sizes"
        ] == MINI_BATCH_SIZES,

        train_dataset["iters"]
        == STAGE_ITERS,

        train_opt[
            "output_kd_opt"
        ]["loss_weight"]
        == experiment["alpha"],

        Path(
            path_opt[
                "pretrain_network_g"
            ]
        ).resolve()
        == STAGE1_G.resolve(),

        Path(
            path_opt[
                "pretrain_network_le"
            ]
        ).resolve()
        == STAGE1_LE.resolve(),
    ]

    if experiment["beta"] is None:
        checks.append(
            "wavelet_kd_opt"
            not in train_opt
        )
    else:
        wavelet = train_opt[
            "wavelet_kd_opt"
        ]

        checks.extend(
            [
                wavelet["loss_weight"]
                == experiment["beta"],

                wavelet["ll_weight"]
                == 1.0,

                wavelet["lh_weight"]
                == experiment["hf"],

                wavelet["hl_weight"]
                == experiment["hf"],

                wavelet["hh_weight"]
                == experiment["hf"],
            ]
        )

    if not all(checks):
        raise RuntimeError(
            "Generated config failed "
            f"verification: {experiment['name']}"
        )


outputkd_base = load_yaml(
    OUTPUTKD_SOURCE
)

wavelet_base = load_yaml(
    WAVELET_SOURCE
)


if not STAGE1_G.exists():
    raise FileNotFoundError(STAGE1_G)

if not STAGE1_LE.exists():
    raise FileNotFoundError(STAGE1_LE)


for experiment in EXPERIMENTS:
    if experiment["beta"] is None:
        config = deepcopy(
            outputkd_base
        )
    else:
        config = deepcopy(
            wavelet_base
        )

    configure_common(
        config,
        experiment,
    )

    verify_config(
        config,
        experiment,
    )

    destination = (
        TRAIN_DIR
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
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )

    print("Created:", destination)
    print(
        "  Model:",
        experiment["model_type"],
    )
    print(
        "  Alpha:",
        experiment["alpha"],
    )
    print(
        "  Beta:",
        experiment["beta"],
    )
    print(
        "  HF:",
        experiment["hf"],
    )


print()
print("All formal Stage 2 configs created.")
print("Total iterations per config:", TOTAL_ITER)
print("GT sizes:", GT_SIZES)
print("Mini-batch sizes:", MINI_BATCH_SIZES)
print("Stage iterations:", STAGE_ITERS)
