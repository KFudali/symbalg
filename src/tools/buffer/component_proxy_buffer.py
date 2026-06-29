import numpy as np
from .value_buffer import ValueBuffer


class ComponentProxyValueBuffer(ValueBuffer):
    def __init__(self, source: ValueBuffer, comp_query: tuple[slice | int, ...]):
        self._source = source
        self._comp_query = comp_query
        self._shape = source.get(0)[comp_query].shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def saved_steps(self) -> int:
        return self._source.saved_steps

    def get(self, index: int = 0) -> np.ndarray:
        return self._source.get(index)[self._comp_query]

    def set(self, value: np.ndarray):
        raise RuntimeError("Cannot set value on a ProxyValueBuffer (read-only view)")

    def set_saved_steps(self, steps: int):
        raise RuntimeError("Cannot change saved_steps on a ProxyValueBuffer")

    def _advance(self, value: np.ndarray) -> None:
        raise RuntimeError("Cannot advance a ProxyValueBuffer")

    def reset(self):
        pass
