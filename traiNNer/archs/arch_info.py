# TODO refactor
from typing import Any, Literal

ARCHS_WITHOUT_FP16 = {
    "atd_light",
    "atd",
    "dat_2",
    "dat_light",
    "dat_s",
    "dat",
    "dctlsa",
    "drct_l",
    "drct_xl",
    "drct",
    "dwt_s",
    "dwt",
    "elan_light",
    "elan",
    "emt",
    "grl_b",
    "grl_s",
    "grl_t",
    "hat_l",
    "hat_m",
    "hat_s",
    "hit_sir",
    "hit_sng",
    "hit_srf",
    "rcan",
    "rcan_unshuffle",
    "rcan_l",
    "moesr2",
    "rgt_s",
    "rgt",
    "seemore_t",
    "srformer_light",
    "srformer",
    "swin2sr_l",
    "swin2sr_m",
    "swin2sr_s",
    "swinir_l",
    "swinir_m",
    "swinir_s",
}

# These archs perform slower when using channels last
ARCHS_WITHOUT_CHANNELS_LAST = {
    "dctlsa",
    "ditn_real",
    "dwt",
    "dwt_s",
    "eimn_a",
    "eimn_l",
    "elan_light",
    "elan",
    "emt",
    "flexnet",
    "grl_b",
    "grl_s",
    "hit_sir",
    "hit_sng",
    "hit_srf",
    "man_light",
    "man_tiny",
    "man",
    "moesr2",
    "omnisr",
    "rgt_s",
    "rgt",
    "rtmosr_l",
    "rtmosr_ul",
    "rtmosr",
    "safmn_l",
    "safmn",
    "seemore_t",
    "spanplus_s",
    "spanplus",
}

# Urban100
OFFICIAL_METRICS: dict[
    str,
    dict[
        int, dict[Literal["df2k_psnr", "df2k_ssim", "div2k_psnr", "div2k_ssim"], float]
    ],
] = {
    "artcnn_r8f48": {
        2: {"df2k_psnr": 31.82, "df2k_ssim": 0.9266},
    },
    "artcnn_r8f64": {
        2: {"df2k_psnr": 32.1, "df2k_ssim": 0.9293},
    },
    "artcnn_r16f96": {
        2: {"df2k_psnr": 32.81, "df2k_ssim": 0.9358},
    },
    "atd": {
        2: {"df2k_psnr": 34.73, "df2k_ssim": 0.9476},
        3: {"df2k_psnr": 30.52, "df2k_ssim": 0.8924},
        4: {"df2k_psnr": 28.22, "df2k_ssim": 0.8414},
    },
    "atd_light": {
        2: {"df2k_psnr": 33.27, "df2k_ssim": 0.9375},
        3: {"df2k_psnr": 29.17, "df2k_ssim": 0.8709},
        4: {"df2k_psnr": 26.97, "df2k_ssim": 0.8107},
    },
    "cfsr": {
        2: {"df2k_psnr": 32.28, "df2k_ssim": 0.9300},
        3: {"df2k_psnr": 28.29, "df2k_ssim": 0.8553},
        4: {"df2k_psnr": 26.21, "df2k_ssim": 0.7897},
    },
    "compact": {
        2: {"df2k_psnr": 31.72, "df2k_ssim": 0.9257},
    },
    "craft": {
        2: {"df2k_psnr": 32.86, "df2k_ssim": 0.9343},
        3: {"df2k_psnr": 28.77, "df2k_ssim": 0.8635},
        4: {"df2k_psnr": 26.56, "df2k_ssim": 0.7995},
    },
    "dat": {
        2: {"df2k_psnr": 34.37, "df2k_ssim": 0.9458},
        3: {"df2k_psnr": 30.18, "df2k_ssim": 0.8886},
        4: {"df2k_psnr": 27.87, "df2k_ssim": 0.8343},
    },
    "dat_s": {
        2: {"df2k_psnr": 34.12, "df2k_ssim": 0.9444},
        3: {"df2k_psnr": 29.98, "df2k_ssim": 0.8846},
        4: {"df2k_psnr": 27.68, "df2k_ssim": 0.8300},
    },
    "dat_light": {
        2: {"df2k_psnr": 32.89, "df2k_ssim": 0.9346},
        3: {"df2k_psnr": 28.89, "df2k_ssim": 0.8666},
        4: {"df2k_psnr": 26.64, "df2k_ssim": 0.8033},
    },
    "dat_2": {
        2: {"df2k_psnr": 34.31, "df2k_ssim": 0.9457},
        3: {"df2k_psnr": 30.13, "df2k_ssim": 0.8878},
        4: {"df2k_psnr": 27.86, "df2k_ssim": 0.8341},
    },
    "dctlsa": {
        2: {"div2k_psnr": 32.96, "div2k_ssim": 0.9362},
        3: {"div2k_psnr": 28.78, "div2k_ssim": 0.8650},
        4: {"div2k_psnr": 26.70, "div2k_ssim": 0.8045},
    },
    "ditn_real": {
        2: {"div2k_psnr": 31.96, "div2k_ssim": 0.9273},
        3: {"div2k_psnr": 28.06, "div2k_ssim": 0.8512},
        4: {"div2k_psnr": 25.99, "div2k_ssim": 0.7837},
    },
    "drct": {
        2: {"df2k_psnr": 34.54, "df2k_ssim": 0.9474},
        3: {"df2k_psnr": 30.34, "df2k_ssim": 0.8910},
        4: {"df2k_psnr": 28.06, "df2k_ssim": 0.8378},
    },
    "drct_l": {
        2: {"df2k_psnr": 35.17, "df2k_ssim": 0.9516},
        3: {"df2k_psnr": 31.14, "df2k_ssim": 0.9004},
        4: {"df2k_psnr": 28.70, "df2k_ssim": 0.8508},
    },
    "dwt": {
        2: {"df2k_psnr": 34.14, "df2k_ssim": 0.9444},
        3: {"df2k_psnr": 30.07, "df2k_ssim": 0.8860},
        4: {"df2k_psnr": 27.81, "df2k_ssim": 0.8324},
    },
    "dwt_s": {
        2: {"df2k_psnr": 33.77, "df2k_ssim": 0.9419},
        3: {"df2k_psnr": 29.73, "df2k_ssim": 0.8806},
        4: {"df2k_psnr": 27.50, "df2k_ssim": 0.8253},
    },
    "eimn_a": {
        2: {"df2k_psnr": 33.15, "df2k_ssim": 0.9373},
        3: {"df2k_psnr": 28.87, "df2k_ssim": 0.8660},
        4: {"df2k_psnr": 26.68, "df2k_ssim": 0.8027},
    },
    "eimn_l": {
        2: {"df2k_psnr": 33.23, "df2k_ssim": 0.9381},
        3: {"df2k_psnr": 29.05, "df2k_ssim": 0.8698},
        4: {"df2k_psnr": 26.88, "df2k_ssim": 0.8084},
    },
    "elan": {
        2: {"div2k_psnr": 33.34, "div2k_ssim": 0.9391},
        3: {"div2k_psnr": 29.32, "div2k_ssim": 0.8745},
        4: {"div2k_psnr": 27.13, "div2k_ssim": 0.8167},
    },
    "elan_light": {
        2: {"div2k_psnr": 32.76, "div2k_ssim": 0.9340},
        3: {"div2k_psnr": 28.69, "div2k_ssim": 0.8624},
        4: {"div2k_psnr": 26.54, "div2k_ssim": 0.7982},
    },
    "emt": {
        3: {"df2k_psnr": 29.16, "df2k_ssim": 0.8716},
        4: {"df2k_psnr": 26.98, "df2k_ssim": 0.8118},
    },
    "esrgan use_pixel_unshuffle=True": {
        4: {
            "df2k_psnr": 27.03,
            "df2k_ssim": 0.8153,
            "div2k_psnr": 26.73,
            "div2k_ssim": 0.8072,
        },
        2: {
            "df2k_psnr": 33.08,
            "df2k_ssim": 0.9387,
        },
        # 2 pixel unshuffle: 33.08   0.9387    (batch 32 lqcrop128 1m iter)
        # 2 not pixel unshuffle: 33.41   0.9407    (batch 16 lqcrop64 1m iter)
    },
    "esrgan use_pixel_unshuffle=False": {
        2: {
            "df2k_psnr": 33.41,
            "df2k_ssim": 0.9407,
        }
        # 2 pixel unshuffle: 33.08   0.9387    (batch 32 lqcrop128 1m iter)
        # 2 not pixel unshuffle: 33.41   0.9407    (batch 16 lqcrop64 1m iter)
    },
    "grl_b": {
        2: {"df2k_psnr": 35.06, "df2k_ssim": 0.9505},
        4: {"df2k_psnr": 28.53, "df2k_ssim": 0.8504},
    },
    "grl_s": {
        2: {"df2k_psnr": 34.36, "df2k_ssim": 0.9463},
        4: {"df2k_psnr": 27.90, "df2k_ssim": 0.8357},
    },
    "grl_t": {
        2: {"df2k_psnr": 33.60, "df2k_ssim": 0.9411},
        4: {"df2k_psnr": 27.15, "df2k_ssim": 0.8185},
    },
    "hat_s": {
        2: {"df2k_psnr": 34.31, "df2k_ssim": 0.9459},
        3: {"df2k_psnr": 30.15, "df2k_ssim": 0.8879},
        4: {"df2k_psnr": 27.87, "df2k_ssim": 0.8346},
    },
    "hat_m": {
        2: {"df2k_psnr": 34.45, "df2k_ssim": 0.9466},
        3: {"df2k_psnr": 30.23, "df2k_ssim": 0.8896},
        4: {"df2k_psnr": 27.97, "df2k_ssim": 0.8368},
    },
    "hat_l": {
        2: {"df2k_psnr": 35.09, "df2k_ssim": 0.9513},
        3: {"df2k_psnr": 30.92, "df2k_ssim": 0.8981},
        4: {"df2k_psnr": 28.60, "df2k_ssim": 0.8498},
    },
    "hit_srf": {
        2: {"div2k_psnr": 33.13, "div2k_ssim": 0.9372},
        3: {"div2k_psnr": 28.99, "div2k_ssim": 0.8687},
        4: {"div2k_psnr": 26.80, "div2k_ssim": 0.8069},
    },
    "hit_sng": {
        2: {"div2k_psnr": 33.01, "div2k_ssim": 0.9360},
        3: {"div2k_psnr": 28.91, "div2k_ssim": 0.8671},
        4: {"div2k_psnr": 26.75, "div2k_ssim": 0.8053},
    },
    "hit_sir": {
        2: {"div2k_psnr": 33.02, "div2k_ssim": 0.9365},
        3: {"div2k_psnr": 28.93, "div2k_ssim": 0.8673},
        4: {"div2k_psnr": 26.71, "div2k_ssim": 0.8045},
    },
    "lmlt_base": {
        2: {"df2k_psnr": 32.52, "df2k_ssim": 0.9316},
        3: {"df2k_psnr": 28.48, "df2k_ssim": 0.8581},
        4: {"df2k_psnr": 26.44, "df2k_ssim": 0.7949},
    },
    "lmlt_large": {
        2: {"df2k_psnr": 32.75, "df2k_ssim": 0.9336},
        3: {"df2k_psnr": 28.72, "df2k_ssim": 0.8628},
        4: {"df2k_psnr": 26.63, "df2k_ssim": 0.8001},
    },
    "lmlt_tiny": {
        2: {"df2k_psnr": 32.04, "df2k_ssim": 0.9273},
        3: {"df2k_psnr": 28.10, "df2k_ssim": 0.8503},
        4: {"df2k_psnr": 26.08, "df2k_ssim": 0.7838},
    },
    "man": {
        2: {"df2k_psnr": 33.73, "df2k_ssim": 0.9422},
        3: {"df2k_psnr": 29.52, "df2k_ssim": 0.8782},
        4: {"df2k_psnr": 27.26, "df2k_ssim": 0.8197},
    },
    "man_tiny": {
        4: {"df2k_psnr": 25.84, "df2k_ssim": 0.7786},
    },
    "man_light": {
        4: {"df2k_psnr": 26.70, "df2k_ssim": 0.8052},
    },
    "moesr2": {
        4: {"df2k_psnr": 27.05, "df2k_ssim": 0.8177},
    },
    "omnisr": {
        2: {
            "df2k_psnr": 33.30,
            "df2k_ssim": 0.9386,
            "div2k_psnr": 33.05,
            "div2k_ssim": 0.9363,
        },
        3: {
            "df2k_psnr": 29.12,
            "df2k_ssim": 0.8712,
            "div2k_psnr": 28.84,
            "div2k_ssim": 0.8656,
        },
        4: {
            "df2k_psnr": 26.95,
            "df2k_ssim": 0.8105,
            "div2k_psnr": 26.64,
            "div2k_ssim": 0.8018,
        },
    },
    "plksr": {
        2: {
            "df2k_psnr": 33.36,
            "df2k_ssim": 0.9395,
            "div2k_psnr": 32.99,
            "div2k_ssim": 0.9365,
        },
        3: {
            "df2k_psnr": 29.10,
            "df2k_ssim": 0.8713,
            "div2k_psnr": 28.86,
            "div2k_ssim": 0.8666,
        },
        4: {
            "df2k_psnr": 26.85,
            "df2k_ssim": 0.8097,
            "div2k_psnr": 26.69,
            "div2k_ssim": 0.8054,
        },
    },
    "plksr_tiny": {
        2: {
            "df2k_psnr": 32.58,
            "df2k_ssim": 0.9328,
            "div2k_psnr": 32.43,
            "div2k_ssim": 0.9314,
        },
        3: {
            "df2k_psnr": 28.51,
            "df2k_ssim": 0.8599,
            "div2k_psnr": 28.35,
            "div2k_ssim": 0.8571,
        },
        4: {
            "df2k_psnr": 26.34,
            "df2k_ssim": 0.7942,
            "div2k_psnr": 26.12,
            "div2k_ssim": 0.7888,
        },
    },
    "realplksr pixelshuffle layer_norm=True": {
        2: {
            "df2k_psnr": 33.44,
            "df2k_ssim": 0.9412,
        },
        4: {
            "df2k_psnr": 26.94,
            "df2k_ssim": 0.8140,
        },
    },
    "rcan": {
        2: {
            "div2k_psnr": 33.34,
            "div2k_ssim": 0.9384,
            "df2k_psnr": 33.62,
            "df2k_ssim": 0.9410,
        },
        3: {"div2k_psnr": 29.09, "div2k_ssim": 0.8702},
        4: {
            "div2k_psnr": 26.82,
            "div2k_ssim": 0.8087,
            "df2k_psnr": 27.16,
            "df2k_ssim": 0.8168,
        },
    },
    "rgt": {
        2: {"df2k_psnr": 34.47, "df2k_ssim": 0.9467},
        3: {"df2k_psnr": 30.28, "df2k_ssim": 0.8899},
        4: {"df2k_psnr": 27.98, "df2k_ssim": 0.8369},
    },
    "rgt_s": {
        2: {"df2k_psnr": 34.32, "df2k_ssim": 0.9457},
        3: {"df2k_psnr": 30.18, "df2k_ssim": 0.8884},
        4: {"df2k_psnr": 27.89, "df2k_ssim": 0.8347},
    },
    "safmn": {
        2: {"df2k_psnr": 31.84, "df2k_ssim": 0.9256},
        3: {"df2k_psnr": 27.95, "df2k_ssim": 0.8474},
        4: {"df2k_psnr": 25.97, "df2k_ssim": 0.7809},
    },
    "safmn_l": {
        2: {"df2k_psnr": 33.06, "df2k_ssim": 0.9366},
        3: {"df2k_psnr": 28.99, "df2k_ssim": 0.8679},
        4: {"df2k_psnr": 26.81, "df2k_ssim": 0.8058},
    },
    "seemore_t": {
        2: {"df2k_psnr": 32.22, "df2k_ssim": 0.9286},
        3: {"df2k_psnr": 28.27, "df2k_ssim": 0.8538},
        4: {"df2k_psnr": 26.23, "df2k_ssim": 0.7883},
    },
    "span": {
        2: {"df2k_psnr": 32.24, "df2k_ssim": 0.9294},
        4: {"df2k_psnr": 26.18, "df2k_ssim": 0.7879},
    },
    "span_s": {
        2: {"df2k_psnr": 32.20, "df2k_ssim": 0.9288},
        4: {"df2k_psnr": 26.13, "df2k_ssim": 0.7865},
    },
    "srformer": {
        2: {"df2k_psnr": 34.09, "df2k_ssim": 0.9449},
        3: {"df2k_psnr": 30.04, "df2k_ssim": 0.8865},
        4: {"df2k_psnr": 27.68, "df2k_ssim": 0.8311},
    },
    "srformer_light": {
        2: {"df2k_psnr": 32.91, "df2k_ssim": 0.9353},
        3: {"df2k_psnr": 28.81, "df2k_ssim": 0.8655},
        4: {"df2k_psnr": 26.67, "df2k_ssim": 0.8032},
    },
    "swinir_s": {
        2: {"df2k_psnr": 32.76, "df2k_ssim": 0.9340},
        3: {"df2k_psnr": 28.66, "df2k_ssim": 0.8624},
        4: {"df2k_psnr": 26.47, "df2k_ssim": 0.7980},
    },
    "swinir_m": {
        2: {"df2k_psnr": 33.81, "df2k_ssim": 0.9427},
        3: {"df2k_psnr": 29.75, "df2k_ssim": 0.8826},
        4: {"df2k_psnr": 27.45, "df2k_ssim": 0.8254},
    },
    "swin2sr_m": {
        2: {"df2k_psnr": 33.89, "df2k_ssim": 0.9431},
        4: {"df2k_psnr": 27.51, "df2k_ssim": 0.8271},
    },
    "swin2sr_s": {
        2: {"df2k_psnr": 32.85, "df2k_ssim": 0.9349},
    },
}

OFFICIAL_SETTINGS_FROMSCRATCH: dict[str, dict[str, Any]] = {
    "atd": {
        "milestones": [250000],
        "total_iter": 300000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "atd_light": {
        "milestones": [250000, 400000, 450000, 475000, 490000],
        "total_iter": 500000,
        "warmup_iter": 20000,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 2",
    },
    "compact": {
        "milestones": [100000, 200000, 300000, 400000, 425000],
        "total_iter": 450000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # recommended: 64",
        "accum_iter": 1,
    },
    "dat": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "dwt": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "dwt_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "ditn_real": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": 1,
    },
    "elan": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": "64  # paper: 48",
        "batch_size_per_gpu": "8  # paper: 32",
        "accum_iter": 1,
    },
    "elan_light": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": "64",
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": 1,
    },
    "rgt": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "plksr": {
        "milestones": [100000, 200000, 300000, 400000, 425000],
        "total_iter": 450000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "span": {
        "milestones": [200000, 400000, 600000, 800000],
        "total_iter": 1000000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "esrgan": {
        "milestones": [200000, 400000, 600000, 800000],
        "total_iter": 1000000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": "64  # paper: 32",
        "batch_size_per_gpu": "8",
        "accum_iter": "1  # paper: 2",
    },
    "omnisr": {
        "milestones": [200000, 400000, 600000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": "1",
    },
    "man": {
        "milestones": [800000, 1200000, 140000, 1500000],
        "total_iter": 1600000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct_l": {
        "milestones": [300000, 500000, 650000, 700000, 750000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hit_srf": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hat_l": {
        "milestones": [300000, 500000, 650000, 700000, 750000],
        "total_iter": 800000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "hat_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "rcan": {
        "milestones": [100000, 200000, 300000, 400000, 450000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # recommended: 32 or 64",
        "accum_iter": "1",
    },
    "seemore_t": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 32",
        "accum_iter": 1,
    },
    "safmn": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": 1,
    },
    "safmn_l": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 48",
        "accum_iter": 1,
    },
    "srformer": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_m": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_m": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_s": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "": {
        "milestones": [250000, 400000, 450000, 475000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1",
    },
}

OFFICIAL_SETTINGS_FINETUNE: dict[str, dict[str, Any]] = {
    "atd": {
        "milestones": [150000, 200000, 225000, 240000],
        "total_iter": 250000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 96,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "atd_light": {
        "milestones": [250000, 400000, 450000, 475000, 490000],
        "total_iter": 500000,
        "warmup_iter": 10000,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 2",
    },
    "dat": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "ditn_real": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": 1,
    },
    "dwt": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "dwt_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "elan": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": "64  # paper: 48",
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "elan_light": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "rgt": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "plksr": {
        "milestones": [100000],
        "total_iter": 50000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "span": {
        "milestones": [100000, 200000, 300000, 400000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4  # paper: !!float 2.5e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "16  # paper: 64",
        "accum_iter": "1",
    },
    "esrgan": {
        "milestones": [100000, 200000, 300000, 400000],
        "total_iter": 500000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": "64  # paper: 32",
        "batch_size_per_gpu": "8",
        "accum_iter": "1  # paper: 2",
    },
    "omnisr": {
        "milestones": [100000, 200000, 300000],
        "total_iter": 400000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": "8  # paper: 64",
        "accum_iter": "1",
    },
    "man": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "drct_l": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hit_srf": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "hat_l": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "hat_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 4,
        "accum_iter": "1  # paper: 8",
    },
    "rcan": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "safmn": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "safmn_l": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "seemore_t": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": 1,
    },
    "srformer": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_m": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swinir_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_m": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1  # paper: 4",
    },
    "swin2sr_s": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 2e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 16,
        "accum_iter": "1  # paper: 4",
    },
    "compact": {
        "milestones": [50000, 100000, 150000, 200000, 225000],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 96,
        "batch_size_per_gpu": "8",
        "accum_iter": 1,
    },
    "": {
        "milestones": [125000, 200000, 225000, 237500],
        "total_iter": 250000,
        "warmup_iter": -1,
        "lr": "!!float 1e-4",
        "lq_size": 64,
        "batch_size_per_gpu": 8,
        "accum_iter": "1",
    },
}


def initialize_official_settings(settings: dict[str, dict[str, Any]]) -> None:
    settings["ultracompact"] = settings["compact"]
    settings["superultracompact"] = settings["compact"]

    settings["dat_2"] = settings["dat"]
    settings["dat_s"] = settings["dat"]
    settings["dat_light"] = settings["dat"]

    settings["drct_xl"] = settings["drct_l"]

    settings["esrgan_lite"] = settings["esrgan"]

    settings["hat_m"] = settings["hat_l"]

    settings["hit_sng"] = settings["hit_srf"]
    settings["hit_sir"] = settings["hit_srf"]

    settings["man_light"] = settings["man"]
    settings["man_tiny"] = settings["man"]

    settings["plksr_tiny"] = settings["plksr"]
    settings["realplksr"] = settings["plksr"]

    settings["rgt_s"] = settings["rgt"]

    settings["spanplus"] = settings["span"]
    settings["spanplus_sts"] = settings["span"]
    settings["spanplus_st"] = settings["span"]
    settings["spanplus_s"] = settings["span"]

    settings["srformer_light"] = settings["srformer"]

    settings["swin2sr_l"] = settings["swin2sr_m"]

    settings["swinir_l"] = settings["swinir_m"]


initialize_official_settings(OFFICIAL_SETTINGS_FROMSCRATCH)
initialize_official_settings(OFFICIAL_SETTINGS_FINETUNE)
