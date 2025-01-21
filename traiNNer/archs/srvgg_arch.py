import torch.nn.functional as F  # noqa: N812
from spandrel.util import store_hyperparameters
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
from torch.nn import init, Module



class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ):
        super().__init__()

        try:
            assert in_channels >= groups and in_channels % groups == 0
        except:  # noqa: E722
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)  # noqa: B904

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .float()
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H],
            dtype=torch.float32,
            device=x.device,
            pin_memory=False,  # pin_memory was originally True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
            .float()
        )
        output = (
            torch.ops.aten.grid_sampler_2d(
                x.reshape(B * self.groups, -1, H, W).float(), coords.float(), 0, 1, True
            )
            .to(x.dtype)
            .view(B, -1, self.scale * H, self.scale * W)
        )

        if self.end_convolution:
            output = self.end_conv(output.to(x.dtype)).to(x.dtype)

        return output.to(x.dtype)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


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

@ARCH_REGISTRY.register()
class SRVGGNetCompact_Dynamic(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="mish",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(DynamicConvolution(3,1,3, num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "mish":
                activation = nn.Mish(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)  # type: ignore

        # the body structure
        for _ in range(num_conv):
            self.body.append(DynamicConvolution(3,1,num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "mish":
                activation = nn.Mish(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)  # type: ignore

        # the last conv
        #self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = DySample(self.num_feat, self.num_feat, upscale)
        self.dynamic_prio = DynamicConvolution(
            3,
            1,
            in_channels=self.num_feat,
            out_channels=self.num_feat,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.dynamic = DynamicConvolution(
            3,
            1,
            in_channels=self.num_feat,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=True,
        )


    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.dynamic_prio(out)
        out = self.upsampler(out)
        out = self.dynamic(out)

        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out


@ARCH_REGISTRY.register()
def dynamiccompact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 16,
    scale: int = 4,
    act_type: str = "mish",
) -> SRVGGNetCompact_Dynamic:
    return SRVGGNetCompact_Dynamic(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
    )


def test():
    model = SRVGGNetCompact(upscale=2)
    test_input = torch.randn(1, 3, 32, 32)
    out = model(test_input)
    print(out.shape)

if __name__ == "__main__":
    test()



@store_hyperparameters()
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    hyperparameters = {}  # noqa: RUF012

    def __init__(
        self,
        *,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
        learn_residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.learn_residual = learn_residual

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)  # type: ignore

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)  # type: ignore

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for i in range(len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)

        if self.learn_residual:
            # add the nearest upsampled image, so that the network learns the residual
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            out += base

        return out


@ARCH_REGISTRY.register()
def compact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 16,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
    )


@ARCH_REGISTRY.register()
def ultracompact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 64,
    num_conv: int = 8,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
    )


@ARCH_REGISTRY.register()
def superultracompact(
    num_in_ch: int = 3,
    num_out_ch: int = 3,
    num_feat: int = 24,
    num_conv: int = 8,
    scale: int = 4,
    act_type: str = "prelu",
    learn_residual: bool = True,
) -> SRVGGNetCompact:
    return SRVGGNetCompact(
        upscale=scale,
        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        num_feat=num_feat,
        num_conv=num_conv,
        act_type=act_type,
        learn_residual=learn_residual,
    )
