from collections.abc import Sequence
from typing import Literal

from spandrel.architectures.HAT import HAT

from traiNNer.utils.registry import SPANDREL_REGISTRY


@SPANDREL_REGISTRY.register()
def hat_l(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
) -> HAT:
    return HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )


@SPANDREL_REGISTRY.register()
def hat_m(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 180,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 3,
    squeeze_factor: float = 30,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
) -> HAT:
    return HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )


@SPANDREL_REGISTRY.register()
def hat_s(
    scale: int = 4,
    img_size: int = 64,
    patch_size: int = 1,
    in_chans: int = 3,
    embed_dim: int = 144,
    depths: Sequence[int] = (6, 6, 6, 6, 6, 6),
    num_heads: Sequence[int] = (6, 6, 6, 6, 6, 6),
    window_size: int = 16,
    compress_ratio: float = 24,
    squeeze_factor: float = 24,
    conv_scale: float = 0.01,
    overlap_ratio: float = 0.5,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    qk_scale: float | None = None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    ape: bool = False,
    patch_norm: bool = True,
    use_checkpoint: bool = False,
    img_range: float = 1.0,
    upsampler: Literal["pixelshuffle"] = "pixelshuffle",
    resi_connection: str = "1conv",
    num_feat: int = 64,
) -> HAT:
    return HAT(
        upscale=scale,
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        compress_ratio=compress_ratio,
        squeeze_factor=squeeze_factor,
        conv_scale=conv_scale,
        overlap_ratio=overlap_ratio,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        ape=ape,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=resi_connection,
        num_feat=num_feat,
    )
