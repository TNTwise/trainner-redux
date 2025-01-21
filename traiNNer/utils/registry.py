# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py


from collections.abc import Callable, Iterator, KeysView
from typing import overload


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map: dict[str, Callable] = {}

    def _do_register(self, name: str, obj: Callable, suffix: str | None = None) -> None:
        if isinstance(suffix, str):
            name = name + "_" + suffix

        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    @overload
    def register(self, obj: None = None, suffix: str | None = None) -> Callable: ...

    @overload
    def register(self, obj: Callable, suffix: str | None = None) -> None: ...

    def register(
        self, obj: Callable | None = None, suffix: str | None = None
    ) -> Callable | None:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Callable) -> Callable:
                name = func_or_class.__name__.lower()
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__.lower()
        self._do_register(name, obj, suffix)

    def get(self, name: str, suffix: str = "traiNNer") -> Callable:
        name = name.lower()
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + "_" + suffix)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._obj_map

    def __iter__(self) -> Iterator[tuple[str, Callable]]:
        return iter(self._obj_map.items())

    def keys(self) -> KeysView[str]:
        return self._obj_map.keys()


DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry("arch")
SPANDREL_REGISTRY = Registry("spandrel")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
