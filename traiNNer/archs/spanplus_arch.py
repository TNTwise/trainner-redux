# https://github.com/umzi2/SPANPlus/blob/master/neosr/archs/spanplus_arch.py

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from spandrel.architectures.__arch_helpers.dysample import DySample
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair
from torch.nn import init
import math


from traiNNer.utils.registry import ARCH_REGISTRY

upsampler_type = Literal["dys", "lp", "ps", "conv"]

class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer



class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, nof_kernels),
        )

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)

class DynamicConvolution(TempModule):
    def __init__(
        self,
        nof_kernels,
        reduce,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: size of the kernel.
        :param groups: controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups
        self.conv_args = {"stride": stride, "padding": padding, "dilation": dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(
            in_channels, max(1, in_channels // reduce), nof_kernels
        )
        self.kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(
            torch.Tensor(
                nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.kernels_bias = nn.Parameter(
                torch.Tensor(nof_kernels, out_channels), requires_grad=True
            )
        else:
            self.register_parameter("kernels_bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x, temperature=1):
        batch_size = x.shape[0]

        alphas = self.attention(x, temperature)
        agg_weights = torch.sum(
            torch.mul(
                self.kernels_weights.unsqueeze(0),
                alphas.view(batch_size, -1, 1, 1, 1, 1),
            ),
            dim=1,
        )
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(
            -1, *agg_weights.shape[-3:]
        )  # batch_size*out_c X in_c X kernel_size X kernel_size
        if self.kernels_bias is not None:
            agg_bias = torch.sum(
                torch.mul(
                    self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)
                ),
                dim=1,
            )
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])  # 1 X batch_size*out_c X H X W

        out = F.conv2d(
            x_grouped,
            agg_weights,
            agg_bias,
            groups=self.groups * batch_size,
            **self.conv_args,
        )  # 1 X batch_size*out_C X H' x W'
        out = out.view(batch_size, -1, *out.shape[-2:])  # batch_size X out_C X H' x W'

        return out

class Conv3XC(nn.Module):
    def __init__(
        self, c_in: int, c_out: int, gain: int = 1, s: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )
        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        if self.training:
            trunc_normal_(self.sk.weight, std=0.02)
            trunc_normal_(self.eval_conv.weight, std=0.02)

        if self.training is False:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # type: ignore
            self.update_params()

    def update_params(self) -> None:
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        W_pixels_to_pad = (target_kernel_size - 1) // 2  # noqa: N806
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat  # type: ignore

    def forward(self, x):  # noqa: ANN201, ANN001
        if self.training:
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
            self.update_params()
        else:
            self.update_params()
            out = self.eval_conv(x)

        return out


class SPAB(nn.Module):
    def __init__(self, in_channels: int, end: bool = False) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c2_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.c3_r = Conv3XC(in_channels, in_channels, gain=2, s=1)
        self.act1 = nn.Mish(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.end = end

    def forward(self, x):  # noqa: ANN201, ANN001
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPABS(nn.Module):
    def __init__(self, feature_channels: int, n_blocks: int = 4, drop: float = 0.0):
        super(SPABS, self).__init__()
        self.block_1 = SPAB(feature_channels)

        self.block_n = nn.Sequential(*[SPAB(feature_channels) for _ in range(n_blocks)])
        self.block_end = SPAB(feature_channels, True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain=2, s=1)
        self.conv_cat = nn.Conv2d(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout2d(drop)
        if self.training:
            trunc_normal_(self.conv_cat.weight, std=0.02)

    def forward(self, x):
        out_b1 = self.block_1(x)
        out_x = self.block_n(out_b1)
        out_end, out_x_2 = self.block_end(out_x)
        out_end = self.dropout(self.conv_2(out_end))
        return self.conv_cat(torch.cat([x, out_end, out_b1, out_x_2], 1))



class SpanPlus(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        blocks: Sequence[int] | None = None,
        feature_channels: int = 48,
        upscale: int = 4,
        drop_rate: float = 0.0,
        upsampler: str = "dys",  # "lp", "ps", "conv"- only 1x
    ) -> None:
        if blocks is None:
            blocks = [4]
        super().__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch if upsampler == "dys" else num_in_ch

        if not self.training:
            drop_rate = 0
        self.feats = nn.Sequential(
            *[Conv3XC(in_channels, feature_channels, gain=2, s=1)]
            + [SPABS(feature_channels, n_blocks, drop_rate) for n_blocks in blocks]
        )
        if upsampler == "ps":
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, out_channels * (upscale**2), 3, padding=1),
                nn.PixelShuffle(upscale),
            )
        elif upsampler == "dys":
            self.upsampler = DySample(feature_channels, feature_channels, upscale)
        elif upsampler == "conv":
            if upscale != 1:
                msg = "conv supports only 1x"
                raise ValueError(msg)

            self.upsampler = nn.Conv2d(feature_channels, out_channels, 3, padding=1)
        else:
            raise NotImplementedError(
                f'upsampler: {upsampler} not supported, choose one of these options: \
                ["ps", "dys", "conv"] conv supports only 1x'
            )
        self.dynamic = DynamicConvolution(
            3,
            1,
            in_channels=feature_channels,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.dynamic_prio = DynamicConvolution(
            3,
            1,
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
    def forward(self, x):  # noqa: ANN201, ANN001
        out = self.feats(x)
        out = self.dynamic_prio(out)
        out = self.upsampler(out)
        out = self.dynamic(out)
        return out


@ARCH_REGISTRY.register()
def spanplus(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    blocks: Sequence[int] | None = None,
    feature_channels: int = 48,
    drop_rate: float = 0.0,
    upsampler: upsampler_type = "dys",  # "lp", "ps", "conv"- only 1x
) -> SpanPlus:
    return SpanPlus(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        blocks=blocks,
        feature_channels=feature_channels,
        upscale=scale,
        drop_rate=drop_rate,
        upsampler=upsampler,
    )


@ARCH_REGISTRY.register()
def spanplus_sts(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    blocks: Sequence[int] | None = (2,),
    feature_channels: int = 32,
    drop_rate: float = 0.0,
    upsampler: upsampler_type = "ps",  # "lp", "ps", "conv"- only 1x
) -> SpanPlus:
    return SpanPlus(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        blocks=blocks,
        feature_channels=feature_channels,
        upscale=scale,
        drop_rate=drop_rate,
        upsampler=upsampler,
    )


@ARCH_REGISTRY.register()
def spanplus_s(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    blocks: Sequence[int] | None = (2,),
    feature_channels: int = 32,
    drop_rate: float = 0.0,
    upsampler: upsampler_type = "dys",  # "lp", "ps", "conv"- only 1x
) -> SpanPlus:
    return SpanPlus(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        blocks=blocks,
        feature_channels=feature_channels,
        upscale=scale,
        drop_rate=drop_rate,
        upsampler=upsampler,
    )


@ARCH_REGISTRY.register()
def spanplus_st(
    scale: int = 4,
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    blocks: Sequence[int] | None = None,
    feature_channels: int = 48,
    drop_rate: float = 0.0,
    upsampler: upsampler_type = "ps",  # "lp", "ps", "conv"- only 1x
) -> SpanPlus:
    return SpanPlus(
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        blocks=blocks,
        feature_channels=feature_channels,
        upscale=scale,
        drop_rate=drop_rate,
        upsampler=upsampler,
    )
