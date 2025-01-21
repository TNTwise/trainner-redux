import os
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models import VGG19_Weights, vgg

# from traiNNer.losses.dists_loss import L2pooling

VGG19_PATCH_SIZE = 256
VGG19_CROP_SIZE = 224
VGG_PRETRAIN_PATH = "experiments/pretrained_models/vgg19-dcbb9e9d.pth"
NAMES = {
    "vgg11": [
        "conv1_1",
        "relu1_1",
        "pool1",
        "conv2_1",
        "relu2_1",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg13": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "pool5",
    ],
    "vgg16": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "pool5",
    ],
    "vgg19": [
        "conv1_1",
        "relu1_1",
        "conv1_2",
        "relu1_2",
        "pool1",
        "conv2_1",
        "relu2_1",
        "conv2_2",
        "relu2_2",
        "pool2",
        "conv3_1",
        "relu3_1",
        "conv3_2",
        "relu3_2",
        "conv3_3",
        "relu3_3",
        "conv3_4",
        "relu3_4",
        "pool3",
        "conv4_1",
        "relu4_1",
        "conv4_2",
        "relu4_2",
        "conv4_3",
        "relu4_3",
        "conv4_4",
        "relu4_4",
        "pool4",
        "conv5_1",
        "relu5_1",
        "conv5_2",
        "relu5_2",
        "conv5_3",
        "relu5_3",
        "conv5_4",
        "relu5_4",
        "pool5",
    ],
}


def insert_bn(names: list[str]) -> list[str]:
    """Insert bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if "conv" in name:
            position = name.replace("conv", "")
            names_bn.append("bn" + position)
    return names_bn


class L2pooling(nn.Module):
    def __init__(
        self,
        channels: int,
        filter_size: int = 5,
        stride: int = 2,
        as_loss: bool = True,
        pad_off: int = 0,
    ) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

    def forward(self, input: Tensor) -> Tensor:
        input = input**2
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],
        )
        return (out + 1e-12).sqrt()


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(
        self,
        layer_name_list: list[str],
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        requires_grad: bool = False,
        remove_pooling: bool = False,
        use_replicate_padding: bool = False,
        pooling_stride: int = 2,
        use_l2_pooling: bool = False,
    ) -> None:
        super().__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        self.names = NAMES[vgg_type.replace("_bn", "")]
        if "bn" in vgg_type:
            self.names = insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            max_idx = max(idx, max_idx)

        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(
                VGG_PRETRAIN_PATH,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(weights=VGG19_Weights.DEFAULT)

        features = vgg_net.features[: max_idx + 1]

        # Reduces edge artifacts
        # https://github.com/crowsonkb/style-transfer-pytorch/blob/e7e2c7134e3937be05ff9f5fcc0873fe5ceb6060/style_transfer/style_transfer.py#L39
        if use_replicate_padding:
            features[0] = self._change_padding_mode(features[0], "replicate")

        modified_net = OrderedDict()
        l2pooling_channels = [64, 128, 256, 512]
        l2pooling_i = 0
        for k, v in zip(self.names, features, strict=False):
            if "pool" in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:  # noqa: PLR5501
                    # in some cases, we may want to change the default stride
                    if use_l2_pooling:
                        if l2pooling_i < len(l2pooling_channels):
                            modified_net[k] = L2pooling(
                                channels=l2pooling_channels[l2pooling_i]
                            )
                        l2pooling_i += 1
                    else:
                        modified_net[k] = nn.MaxPool2d(
                            kernel_size=2, stride=pooling_stride
                        )
            # if "relu" in k:
            #     modified_net[k] = nn.Mish()  # TODO SILU
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        else:
            self.register_buffer("mean", torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    @staticmethod
    def _change_padding_mode(conv: nn.Module, padding_mode: str) -> nn.Conv2d:
        new_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            padding_mode=padding_mode,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            if new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        return new_conv

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")  # pyright: ignore[reportPrivateImportUsage] # https://github.com/pytorch/pytorch/issues/131765
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.range_norm:
            x = (x + 1) / 2

        x = (x - self.mean) / self.std

        output = {}
        for key, layer in self.vgg_net._modules.items():  # noqa: SLF001
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output
