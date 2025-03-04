import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range: int,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std  # type: ignore
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int) -> None:
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(
            torch.zeros((self.out_channels, self.inp_channels, 3, 3)),
            requires_grad=False,
        )
        mask[:, 0 * g : 1 * g, 1, 2] = 1.0
        mask[:, 1 * g : 2 * g, 1, 0] = 1.0
        mask[:, 2 * g : 3 * g, 2, 1] = 1.0
        mask[:, 3 * g : 4 * g, 0, 1] = 1.0
        mask[:, 4 * g :, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x: Tensor) -> Tensor:
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int) -> None:
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.zeros(inp_channels, 1, 3, 3), requires_grad=False
        )
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g : 1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g : 2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g : 3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g : 4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g :, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = F.conv2d(
            input=x,
            weight=self.weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.inp_channels,
        )
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        conv_type: str = "fast-training-speed",
    ) -> None:
        super().__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == "low-training-memory":
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == "fast-training-speed":
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError("invalid type of shift-conv2d")

    def forward(self, x: Tensor) -> Tensor:
        y = self.shift_conv(x)
        return y


class LFE(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        exp_ratio: int = 4,
        act_type: str = "relu",
    ) -> None:
        super().__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("unsupport type of activation")

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y


class GMSA(nn.Module):
    def __init__(
        self,
        channels: int,
        shifts: int = 4,
        window_sizes: Sequence[int] | None = None,
        calc_attn: bool = True,
    ) -> None:
        if window_sizes is None:
            window_sizes = [4, 8, 12]
        super().__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.BatchNorm2d(self.channels * 2),
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns = [channels // 3, channels // 3, channels // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels),
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self, x: Tensor, prev_atns: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        _b, _c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))  # noqa: PLW2901
                q, v = rearrange(
                    x_,
                    "b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c",
                    qv=2,
                    dh=wsize,
                    dw=wsize,
                )
                atn = q @ q.transpose(-2, -1)
                atn = atn.softmax(dim=-1)
                y_ = atn @ v
                y_ = rearrange(
                    y_,
                    "(b h w) (dh dw) c-> b (c) (h dh) (w dw)",
                    h=h // wsize,
                    w=w // wsize,
                    dh=wsize,
                    dw=wsize,
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))  # noqa: PLW2901
                atn = prev_atns[idx]
                v = rearrange(
                    x_, "b (c) (h dh) (w dw) -> (b h w) (dh dw) c", dh=wsize, dw=wsize
                )
                y_ = atn @ v
                y_ = rearrange(
                    y_,
                    "(b h w) (dh dw) c-> b (c) (h dh) (w dw)",
                    h=h // wsize,
                    w=w // wsize,
                    dh=wsize,
                    dw=wsize,
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns


class ELAB(nn.Module):
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        exp_ratio: int = 2,
        shifts: int = 0,
        window_sizes: Sequence[int] | None = None,
        shared_depth: int = 1,
    ) -> None:
        if window_sizes is None:
            window_sizes = [4, 8, 12]
        super().__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe["lfe_0"] = LFE(
            inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio
        )
        modules_gmsa["gmsa_0"] = GMSA(
            channels=inp_channels,
            shifts=shifts,
            window_sizes=window_sizes,
            calc_attn=True,
        )
        for i in range(shared_depth):
            modules_lfe[f"lfe_{i + 1}"] = LFE(
                inp_channels=inp_channels,
                out_channels=out_channels,
                exp_ratio=exp_ratio,
            )
            modules_gmsa[f"gmsa_{i + 1}"] = GMSA(
                channels=inp_channels,
                shifts=shifts,
                window_sizes=window_sizes,
                calc_attn=False,
            )
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x: Tensor) -> Tensor:
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x = self.modules_lfe[f"lfe_{i}"](x) + x
                y, atn = self.modules_gmsa[f"gmsa_{i}"](x, None)
                x = y + x
            else:
                x = self.modules_lfe[f"lfe_{i}"](x) + x
                y, atn = self.modules_gmsa[f"gmsa_{i}"](x, atn)
                x = y + x
        return x


@ARCH_REGISTRY.register()
class ELAN(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        colors: int = 3,
        rgb_range: int = 255,
        norm: bool = False,
        window_sizes: Sequence[int] | None = None,
        m_elan: int = 36,
        c_elan: int = 180,
        n_share: int = 0,
        r_expand: int = 2,
    ) -> None:
        if window_sizes is None:
            window_sizes = [4, 8, 16]
        super().__init__()

        self.scale = scale
        self.colors = colors
        self.window_sizes = window_sizes
        self.m_elan = m_elan
        self.c_elan = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        if norm:
            self.rgb_range = rgb_range
            self.sub_mean = MeanShift(rgb_range)
            self.add_mean = MeanShift(rgb_range, sign=1)
        else:
            self.rgb_range = 1
            self.sub_mean = nn.Identity()
            self.add_mean = nn.Identity()

        # define head module
        m_head = [
            nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)
        ]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1 + self.n_share)):
            if (i + 1) % 2 == 1:
                m_body.append(
                    ELAB(
                        self.c_elan,
                        self.c_elan,
                        self.r_expand,
                        0,
                        self.window_sizes,
                        shared_depth=self.n_share,
                    )
                )
            else:
                m_body.append(
                    ELAB(
                        self.c_elan,
                        self.c_elan,
                        self.r_expand,
                        1,
                        self.window_sizes,
                        shared_depth=self.n_share,
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(
                self.c_elan,
                self.colors * self.scale * self.scale,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(self.scale),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[2:]  # noqa: N806
        x = self.check_image_size(x)

        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)

        return x[:, :, 0 : H * self.scale, 0 : W * self.scale]

    def check_image_size(self, x: Tensor) -> Tensor:
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = (
                wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
            )
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x


@ARCH_REGISTRY.register()
def elan_light(
    scale: int = 4,
    colors: int = 3,
    rgb_range: int = 255,
    norm: bool = False,
    window_sizes: Sequence[int] | None = None,
    m_elan: int = 24,
    c_elan: int = 60,
    n_share: int = 1,
    r_expand: int = 2,
) -> ELAN:
    if window_sizes is None:
        window_sizes = [4, 8, 16]
    return ELAN(
        scale=scale,
        colors=colors,
        rgb_range=rgb_range,
        norm=norm,
        window_sizes=window_sizes,
        m_elan=m_elan,
        c_elan=c_elan,
        n_share=n_share,
        r_expand=r_expand,
    )
