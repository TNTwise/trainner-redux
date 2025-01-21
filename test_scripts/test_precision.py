from typing import Any

import torch
from torch import Tensor, autocast, nn
from traiNNer.archs import ARCH_REGISTRY, SPANDREL_REGISTRY
from traiNNer.losses.basic_loss import L1Loss

ALL_REGISTRIES = list(ARCH_REGISTRY)
EXCLUDE_BENCHMARK_ARCHS = {
    "dat",
    "hat",
    "swinir",
    "lmlt",
    "vggstylediscriminator",
    "unetdiscriminatorsn",
    "vggfeatureextractor",
}

FILTERED_REGISTRY = [
    (name, arch)
    for name, arch in list(SPANDREL_REGISTRY) + list(ARCH_REGISTRY)
    if name not in EXCLUDE_BENCHMARK_ARCHS
]
# For archs that have extra parameters, list all combinations that need to be benchmarked.
EXTRA_ARCH_PARAMS: dict[str, list[dict[str, Any]]] = {
    k: [] for k, _ in FILTERED_REGISTRY
}
EXTRA_ARCH_PARAMS["realplksr"] = [
    {"upsampler": "dysample"},
    {"upsampler": "pixelshuffle"},
]
# A list of tuples in the format of (name, arch, extra_params).
FILTERED_REGISTRIES_PARAMS = [
    (name, arch, extra_params)
    for (name, arch) in FILTERED_REGISTRY
    for extra_params in (EXTRA_ARCH_PARAMS[name] if EXTRA_ARCH_PARAMS[name] else [{}])
]


def format_extra_params(extra_arch_params: dict[str, Any]) -> str:
    out = ""

    for k, v in extra_arch_params.items():
        if isinstance(v, str):
            out += f"{v} "
        else:
            out += f"{k}={v} "

    return out.strip()


def compare_precision(
    net: nn.Module, input_tensor: Tensor, criterion: nn.Module
) -> tuple[float, float]:
    with torch.inference_mode():
        fp32_output = net(input_tensor)

    fp16_loss = None
    try:
        with autocast(dtype=torch.float16, device_type="cuda"), torch.inference_mode():
            fp16_output = net(input_tensor)
        fp16_loss = criterion(fp16_output.float(), fp32_output).item()
    except Exception as e:
        print(f"Error in FP16 inference: {e}")
        fp16_loss = float("inf")

    bf16_loss = None
    try:
        with autocast(dtype=torch.bfloat16, device_type="cuda"), torch.inference_mode():
            bf16_output = net(input_tensor)
        bf16_loss = criterion(bf16_output.float(), fp32_output).item()
    except Exception as e:
        print(f"Error in BF16 inference: {e}")
        bf16_loss = float("inf")

    return fp16_loss, bf16_loss


if __name__ == "__main__":
    scale = 4
    for name, arch, extra_arch_params in FILTERED_REGISTRIES_PARAMS:
        label = f"{name} {format_extra_params(extra_arch_params)} {scale}x"

        try:
            if name not in {
                "rcan",
                "esrgan",
                "compact",
                "span",
                "dat_2",
                "spanplus",
                "realplksr",
            }:
                continue

            net: nn.Module = arch(scale=scale, **extra_arch_params).eval().to("cuda")
            # net.load_state_dict(
            #     torch.load(
            #         r"DAT_2_x4.pth",
            #         weights_only=True,
            #     )["params"]
            # )

            input_tensor = torch.randn((2, 3, 192, 192), device="cuda")
            criterion = L1Loss(1.0)

            fp16_loss, bf16_loss = compare_precision(net, input_tensor, criterion)
            diff = abs(fp16_loss - bf16_loss)

            if fp16_loss < bf16_loss:
                print(
                    f"{label:>30s}: FP16: {fp16_loss:.6f}; BF16: {bf16_loss:.6f}; diff = {diff}"
                )
            else:
                print(
                    f"{label:>30s}: BF16: {bf16_loss:.6f}; FP16: {fp16_loss:.6f}; diff = {diff}"
                )
        except Exception as e:
            print(f"skip {label}", e)
