import sys
from os import path as osp

import torch

from traiNNer.models.realesrgan_model import RealESRGANModel
from traiNNer.utils import RNG
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.registry import MODEL_REGISTRY
from traiNNer.utils.types import DataFeed

OTF_DEBUG_PATH = osp.abspath(
    osp.abspath(osp.join(osp.join(sys.argv[0], osp.pardir), "./debug/otf"))
)

ANTIALIAS_MODES = {"bicubic", "bilinear"}


@MODEL_REGISTRY.register(suffix="traiNNer")
class RealESRGANPairedModel(RealESRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)

        self.dataroot_lq_prob = opt.dataroot_lq_prob

    @torch.no_grad()
    def feed_data(self, data: DataFeed) -> None:
        if RNG.get_rng().uniform() < self.dataroot_lq_prob:
            # paired feed data
            new_data = {
                k.replace("paired_", ""): v
                for k, v in data.items()
                if k.startswith("paired_")
            }

            assert "lq" in new_data
            self.lq = new_data["lq"].to(  # pyright: ignore[reportAttributeAccessIssue]
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            if "gt" in new_data:
                self.gt = new_data["gt"].to(  # pyright: ignore[reportAttributeAccessIssue]
                    self.device,
                    memory_format=self.memory_format,
                    non_blocking=True,
                )

            # moa
            if self.is_train and self.batch_augment and self.gt is not None:
                self.gt, self.lq = self.batch_augment(self.gt, self.lq)  # pyright: ignore[reportArgumentType]

        else:
            # OTF feed data
            new_data = {
                k.replace("otf_", ""): v
                for k, v in data.items()
                if k.startswith("otf_")
            }
            super().feed_data(new_data)  # type: ignore
