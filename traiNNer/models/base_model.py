import json
import os
import time
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytorch_optimizer
import torch
from safetensors.torch import load_file, save_file
from spandrel import ModelLoader, StateDict
from spandrel.architectures.ESRGAN import ESRGAN
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer, ParamsT
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from traiNNer.ops.batchaug import MOA_DEBUG_PATH, BatchAugment
from traiNNer.utils import get_root_logger
from traiNNer.utils.dist_util import master_only
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import is_json_compatible
from traiNNer.utils.options import struct2dict
from traiNNer.utils.redux_options import ReduxOptions
from traiNNer.utils.types import DataFeed, TrainingState


class BaseModel:
    """Base model."""

    def __init__(self, opt: ReduxOptions) -> None:
        self.opt = opt
        self.device = torch.device("cuda" if opt.num_gpu != 0 else "cpu")
        self.is_train = opt.is_train
        self.schedulers: list[LRScheduler] = []
        self.optimizers: list[Optimizer] = []
        self.optimizers_skipped: list[bool] = []
        self.optimizers_schedule_free: list[bool] = []
        self.batch_augment = None
        self.log_dict = {}
        self.loss_samples = 0
        self.with_metrics = (
            opt.val is not None
            and opt.val.val_enabled
            and opt.val.metrics_enabled
            and opt.val.metrics is not None
        )
        self.use_pbar = opt.val is not None and opt.val.pbar
        self.metric_results: dict[str, Any] = {}
        self.best_metric_results: dict[str, Any] = {}
        self.first_val_completed = False
        self.model_loader = ModelLoader()
        self.net_g = None
        self.net_g_ema: AveragedModel | None = None
        self.net_d = None
        self.use_amp = False
        self.use_channels_last = False
        self.memory_format = torch.preserve_format
        self.amp_dtype = torch.float16
        self.scaler_g: GradScaler | None = None
        self.scaler_d: GradScaler | None = None
        self.accum_iters: int = 1
        self.ema_n_averaged: Tensor | None = None
        self.grad_clip: bool = False

    @abstractmethod
    def feed_data(self, data: DataFeed) -> None:
        pass

    @abstractmethod
    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        pass

    @abstractmethod
    def get_current_visuals(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        """Save networks and training state."""

    def validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool = False,
        multi_val_datasets: bool = False,
    ) -> None:
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt.dist:
            self.dist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )
        else:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    @abstractmethod
    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        pass

    @abstractmethod
    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        pass

    def _initialize_best_metric_results(self, dataset_name: str) -> None:
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if dataset_name in self.best_metric_results:
            return

        # add a dataset record
        assert self.opt.val is not None
        assert self.opt.val.metrics is not None
        record = {}
        for metric, content in self.opt.val.metrics.items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = {"better": better, "val": init_val, "iter": -1}
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(
        self, dataset_name: str, metric: str, val: float, current_iter: int
    ) -> None:
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        elif val <= self.best_metric_results[dataset_name][metric]["val"]:
            self.best_metric_results[dataset_name][metric]["val"] = val
            self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def get_current_log(self) -> dict[str, float | torch.Tensor]:
        return {k: v / self.loss_samples for k, v in self.log_dict.items()}

    def reset_current_log(self) -> None:
        self.log_dict = {}
        self.loss_samples = 0

    def model_to_device(self, net: nn.Module) -> nn.Module:
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        assert isinstance(self.opt.num_gpu, int)
        net = net.to(
            self.device,
            memory_format=self.memory_format,
            non_blocking=True,
        )  # pyright: ignore[reportCallIssue] # https://github.com/pytorch/pytorch/issues/131765

        if self.opt.use_compile:
            net = torch.compile(net)  # pyright: ignore[reportAssignmentType]

        if self.opt.dist:
            find_unused_parameters = self.opt.find_unused_parameters
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt.num_gpu > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(
        self,
        optim_type: str,
        params: ParamsT,
        lr: float,
        **kwargs,
    ) -> Optimizer:
        # https://github.com/Corpsecreate/neosr/blob/dbb3d012880add41b6e49d4c5252c719c3236892/neosr/models/default.py#L247
        # uppercase optim_type to make it case insensitive
        optim_type_upper = optim_type.upper()
        optim_map = {
            "ADADELTA": torch.optim.Adadelta,
            "ADAGRAD": torch.optim.Adagrad,
            "ADAM": torch.optim.Adam,
            "ADAMW": torch.optim.AdamW,
            "SPARSEADAM": torch.optim.SparseAdam,
            "ADAMAX": torch.optim.Adamax,
            "ASGD": torch.optim.ASGD,
            "SGD": torch.optim.SGD,
            "RADAM": torch.optim.RAdam,
            "RPROP": torch.optim.Rprop,
            "RMSPROP": torch.optim.RMSprop,
            "NADAM": torch.optim.NAdam,
            "LBFGS": torch.optim.LBFGS,
            "ADAN": pytorch_optimizer.Adan,
            "LAMB": pytorch_optimizer.Lamb,
            "PRODIGY": pytorch_optimizer.Prodigy,
            "LION": pytorch_optimizer.Lion,
            "TIGER": pytorch_optimizer.Tiger,
            "ADAMP": pytorch_optimizer.AdamP,
        }
        if optim_type_upper in optim_map:
            optimizer = optim_map[optim_type_upper](params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supported yet.")
        return optimizer

    def setup_schedulers(self) -> None:
        # https://github.com/Corpsecreate/neosr/blob/a29e509dae5cd39aea94ac82d1347d2a54e1175c/neosr/models/default.py#L276

        """Set up schedulers."""
        assert self.opt.train is not None
        if self.opt.train.scheduler is not None:
            scheduler_opts = struct2dict(self.opt.train.scheduler)
            scheduler_type = scheduler_opts.pop("type")
            # uppercase scheduler_type to make it case insensitive
            sch_typ_upper = scheduler_type.upper()
            sch_map: dict[str, type[LRScheduler]] = {
                "CONSTANTLR": torch.optim.lr_scheduler.ConstantLR,
                "LINEARLR": torch.optim.lr_scheduler.LinearLR,
                "EXPONENTIALLR": torch.optim.lr_scheduler.ExponentialLR,
                "CYCLICLR": torch.optim.lr_scheduler.CyclicLR,
                "STEPLR": torch.optim.lr_scheduler.StepLR,
                "MULTISTEPLR": torch.optim.lr_scheduler.MultiStepLR,
                "LAMBDALR": torch.optim.lr_scheduler.LambdaLR,
                "MULTIPLICATIVELR": torch.optim.lr_scheduler.MultiplicativeLR,
                "SEQUENTIALLR": torch.optim.lr_scheduler.SequentialLR,
                "CHAINEDSCHEDULER": torch.optim.lr_scheduler.ChainedScheduler,
                "ONECYCLELR": torch.optim.lr_scheduler.OneCycleLR,
                "POLYNOMIALLR": torch.optim.lr_scheduler.PolynomialLR,
                "COSINEANNEALINGWARMRESTARTS": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                "COSINEANNEALING": torch.optim.lr_scheduler.CosineAnnealingLR,
                "REDUCELRONPLATEAU": torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
            logger = get_root_logger()
            if sch_typ_upper in sch_map:
                for i, optimizer in enumerate(self.optimizers):
                    if not self.optimizers_schedule_free[i]:
                        self.schedulers.append(
                            sch_map[sch_typ_upper](optimizer, **scheduler_opts)
                        )
                    else:
                        logger.warning(
                            "Scheduler is ignored when using schedule free optimizer."
                        )

            else:
                raise NotImplementedError(
                    f"Scheduler {scheduler_type} is not implemented yet."
                )

    def get_bare_model(
        self, net: DataParallel | DistributedDataParallel | nn.Module
    ) -> nn.Module:
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net = net.module
        return self.unwrap_compiled_model(net)

    def unwrap_compiled_model(
        self, net: nn.Module | torch._dynamo.OptimizedModule
    ) -> nn.Module:
        if isinstance(net, torch._dynamo.OptimizedModule):  # noqa: SLF001
            return net._orig_mod  # noqa: SLF001
        return net

    @master_only
    def print_network(self, net: nn.Module) -> None:
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(x.numel() for x in net.parameters())

        logger = get_root_logger()
        logger.info("Network: %s, with parameters: %s", net_cls_str, f"{net_params:,d}")
        logger.info(net_str)

    def _set_lr(self, lr_groups_l: list[list[float]]) -> None:
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=False):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=False):
                param_group["lr"] = lr

    def _get_init_lr(self) -> list[list[float]]:
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1) -> None:
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int): Warm-up iter numbers. -1 for no warm-up.
                Default: -1.
        """
        for i, scheduler in enumerate(self.schedulers):
            if not self.optimizers_skipped[i]:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self) -> list[float]:
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(
        self,
        net: nn.Module,
        net_label: str,
        save_dir: str,
        current_iter: int,
        param_key: str,
    ) -> None:
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        current_iter_str = "latest" if current_iter == -1 else str(current_iter)
        assert self.opt.logger is not None

        save_filename = (
            f"{net_label}_{current_iter_str}.{self.opt.logger.save_checkpoint_format}"
        )
        save_path = os.path.join(save_dir, save_filename)

        bare_net_ = self.get_bare_model(net)
        state_dict = bare_net_.state_dict()
        new_state_dict = OrderedDict()

        for full_key, param in state_dict.items():
            key = full_key
            if key.startswith("module."):  # remove unnecessary 'module.'
                key = key[7:]
            if key == "n_averaged":  # ema key, breaks compatibility
                continue
            new_state_dict[key] = param.to("cpu", memory_format=torch.contiguous_format)

        metadata: dict[str, Any] | None = None
        if hasattr(net, "hyperparameters"):
            metadata = {
                k: v for k, v in net.hyperparameters.items() if is_json_compatible(v)
            }

        # avoid occasional writing errors
        retry = 3
        logger = None
        while retry > 0:
            try:
                if self.opt.logger.save_checkpoint_format == "safetensors":
                    if metadata:
                        save_file(
                            new_state_dict,
                            save_path,
                            metadata={"metadata": json.dumps(metadata)},
                        )
                    else:
                        save_file(new_state_dict, save_path)
                else:  # noqa: PLR5501
                    if metadata:
                        torch.save(
                            {"metadata": metadata, param_key: new_state_dict}, save_path
                        )
                    else:
                        torch.save(new_state_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    "Save model error: %s, remaining retry times: %d", e, retry - 1
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            assert logger is not None
            logger.warning("Still cannot save %s. Just ignore it.", save_path)
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(
        self,
        crt_net: nn.Module,
        load_net: dict[str, Any],
        file_path: str,
        strict: bool = True,
    ) -> bool:
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net_state_dict = crt_net.state_dict()
        crt_net_keys = set(crt_net_state_dict.keys())
        load_net_keys = set(load_net.keys())
        valid = True

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            current_minus_loaded = crt_net_keys - load_net_keys
            if len(current_minus_loaded) > 0:
                if strict:
                    valid = False
                logger.warning(
                    "Pretrain network is missing %d keys from current network, up to 10 shown:",
                    len(current_minus_loaded),
                )
                for v in sorted(current_minus_loaded)[:10]:
                    logger.warning("    %s", v)
            loaded_minus_current = load_net_keys - crt_net_keys
            if len(loaded_minus_current) > 0:
                if strict:
                    valid = False
                logger.warning(
                    "Current network is missing %d keys from pretrain network, up to 10 shown:",
                    len(loaded_minus_current),
                )
                for v in sorted(loaded_minus_current)[:10]:
                    logger.warning("    %s", v)

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net_state_dict[k].size() != load_net[k].size():
                    logger.warning(
                        "Size different, ignore [bold]%s[/bold]: crt_net: "
                        "%s; load_net: %s",
                        k,
                        crt_net_state_dict[k].shape,
                        load_net[k].shape,
                        extra={"markup": True},
                    )
                    load_net[k + ".ignore"] = load_net.pop(k)

            new_common_keys = crt_net_keys & set(load_net.keys())
            if len(new_common_keys) == 0:
                logger.warning(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model. Pretrain will have no effect and the model will be trained from scratch.",
                    file_path,
                    0,
                )
            elif len(new_common_keys) == len(crt_net_keys):
                logger.info(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model. Pretrain is loaded in strict mode.",
                    file_path,
                    100,
                )
            else:
                overlap = len(new_common_keys) / len(crt_net_keys) * 100
                logger.info(
                    "Pretrain model %s matched %.2f%% of the keys of the currently training model.",
                    file_path,
                    overlap,
                )
                if overlap == 0:
                    valid = False  # even with strict false, reject 0% match as likely user error
        return valid

    def load_network(
        self,
        net: nn.Module,
        load_path: str,
        strict: bool = True,
        param_key: str | None = "params",
    ) -> None:
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()

        # TODO refactor, messy hack to support the different ESRGAN versions
        if isinstance(net, ESRGAN):
            load_net_wrapper = self.model_loader.load_from_file(load_path)
            load_net = load_net_wrapper.model.state_dict()
            valid = self._print_different_keys_loading(net, load_net, load_path, strict)

            if not valid:
                raise ValueError(
                    f"Unable to load pretrain network due to mismatched state dict keys (see above for missing keys): {load_path}"
                )

            net.load_state_dict(load_net, strict=strict)
            logger.info(
                "Loading %s model from %s, with spandrel.",
                net.__class__.__name__,
                clickable_file_path(Path(load_path).absolute().parent, load_path),
            )
        else:
            net = self.get_bare_model(net)
            if load_path.endswith(".safetensors"):
                load_net: StateDict = load_file(load_path, device=str(self.device))
            elif load_path.endswith(".pth"):
                load_net = torch.load(
                    load_path,
                    map_location="cpu",
                    weights_only=True,
                )
            else:
                raise ValueError(f"Unsupported model: {load_path}")

            if param_key is not None:
                if param_key in load_net:
                    load_net = self.remove_common_prefix(load_net[param_key])
                if param_key not in load_net:
                    load_net, new_param_key = self.canonicalize_state_dict(load_net)
                    logger.info(
                        "Loading: %s does not exist, using %s.",
                        param_key,
                        new_param_key,
                    )
                    param_key = new_param_key
            else:
                load_net, param_key = self.canonicalize_state_dict(load_net)

            logger.info(
                "Loading %s model from %s, with param key: [bold]%s[/bold].",
                net.__class__.__name__,
                clickable_file_path(Path(load_path).absolute().parent, load_path),
                param_key,
                extra={"markup": True},
            )

            valid = self._print_different_keys_loading(net, load_net, load_path, strict)

            if not valid:
                raise ValueError(
                    f"Unable to load pretrain network due to mismatched state dict keys (see above for missing keys): {load_path}"
                )

            net.load_state_dict(load_net, strict=strict)

    # https://github.com/chaiNNer-org/spandrel/blob/ebf11bab4bc3fabccc80fcc377eaabb8cecbf8cd/libs/spandrel/spandrel/__helpers/canonicalize.py#L14
    def canonicalize_state_dict(
        self, state_dict: StateDict
    ) -> tuple[StateDict, str | None]:
        """
        Canonicalize a state dict.

        This function is used to canonicalize a state dict, so that it can be
        used for architecture detection and loading.

        This function is not intended to be used in production code.
        """

        used_unwrap_key = None

        # the real state dict might be inside a dict with a known key
        unwrap_keys = [
            "model_state_dict",
            "state_dict",
            "params_ema",
            "params-ema",
            "params",
            "model",
            "net",
        ]
        for unwrap_key in unwrap_keys:
            if unwrap_key in state_dict and isinstance(state_dict[unwrap_key], dict):
                state_dict = state_dict[unwrap_key]
                used_unwrap_key = unwrap_key
                break

        # unwrap single key
        if len(state_dict) == 1:
            single = next(iter(state_dict.values()))
            if isinstance(single, dict):
                state_dict = single

        # remove known common prefixes
        state_dict = self.remove_common_prefix(state_dict, ["module.", "netG."])

        return state_dict, used_unwrap_key

    def remove_common_prefix(
        self, state_dict: StateDict, prefixes: Sequence[str] = ("module.", "netG.")
    ) -> StateDict:
        if len(state_dict) > 0:
            for prefix in prefixes:
                if all(i.startswith(prefix) for i in state_dict.keys()):
                    state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
        return state_dict

    def load_network_spandrel(
        self, net: nn.Module, load_path: str, strict: bool = True
    ) -> None:
        logger = get_root_logger()
        load_net = self.model_loader.load_from_file(load_path)
        net.load_state_dict(load_net.model.state_dict(), strict=strict)
        logger.info(
            "Loading %s model from %s, with spandrel.",
            net.__class__.__name__,
            load_path,
        )

    @master_only
    def save_training_state(
        self,
        epoch: int,
        current_iter: int,
    ) -> None:
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        assert self.opt.path.training_states is not None

        if current_iter != -1:
            assert self.scaler_g is not None
            assert self.scaler_d is not None
            state: TrainingState = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }

            if self.use_amp:
                state["scaler_d"] = self.scaler_d.state_dict()
                state["scaler_g"] = self.scaler_g.state_dict()

            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            if self.net_g_ema is not None:
                state["ema_n_averaged"] = self.net_g_ema.state_dict()["n_averaged"]

            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.opt.path.training_states, save_filename)

            logger = None
            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        "Save training state error: %s, remaining retry times: %d",
                        e,
                        retry - 1,
                    )
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                assert logger is not None
                logger.warning("Still cannot save %s. Just ignore it.", save_path)
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state: TrainingState) -> None:
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        assert self.scaler_d is not None
        assert self.scaler_g is not None

        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]

        assert len(resume_optimizers) == len(self.optimizers), (
            "Wrong lengths of optimizers"
        )
        assert len(resume_schedulers) == len(self.schedulers), (
            "Wrong lengths of schedulers"
        )

        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        if "scaler_g" in resume_state:
            self.scaler_g.load_state_dict(resume_state["scaler_g"])
        if "scaler_d" in resume_state:
            self.scaler_d.load_state_dict(resume_state["scaler_d"])

        if "ema_n_averaged" in resume_state and self.net_g_ema is not None:
            self.net_g_ema.register_buffer("n_averaged", resume_state["ema_n_averaged"])

    def reduce_loss_dict(self, loss_dict: dict[str, Any]) -> OrderedDict[str, Any]:
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt.dist:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)  # type: ignore
                if self.opt.rank == 0:
                    losses /= self.opt.world_size
                loss_dict = dict(zip(keys, losses, strict=False))

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def setup_batchaug(self) -> None:
        assert self.opt.train is not None
        logger = get_root_logger()
        if self.opt.train.use_moa:
            self.batch_augment = BatchAugment(self.opt.scale, self.opt.train)
            logger.info(
                "Mixture of augmentations (MoA) enabled with augs: %s and probs: %s",
                self.batch_augment.moa_augs,
                self.batch_augment.moa_probs,
            )
            if self.batch_augment.debug:
                logger.info(
                    "MoA debugging enabled. Augmented tiles will be saved to: %s",
                    MOA_DEBUG_PATH,
                )
