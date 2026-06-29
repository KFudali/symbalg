import numpy as np
from .value_buffer import ValueBuffer


class ComponentProxyValueBuffer(ValueBuffer):
    def __init__(self, source: ValueBuffer, source_shape: tuple[slice, ...]):
        self._source = source
        self._source_shape = source_shape
        sliced = source.get(0)[source_shape]
        self._squeeze_axes = tuple(
            i
            for i, s in enumerate(source_shape)
            if isinstance(s, slice) and sliced.shape[i] == 1
        )
        self._shape = np.squeeze(sliced, axis=self._squeeze_axes).shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def saved_steps(self) -> int:
        return self._source.saved_steps

    def get(self, index: int = 0) -> np.ndarray:
        if index < 0:
            raise IndexError("index must be >= 0")
        if index >= self.saved_steps:
            raise IndexError(f"Requested index {index} exceeds available past history")
        sliced = self._source.get(index)[self._source_shape]
        if self._squeeze_axes:
            return np.squeeze(sliced, axis=self._squeeze_axes)
        return sliced

    def set(self, value: np.ndarray):
        raise RuntimeError("Cannot set value on a ProxyValueBuffer (read-only view)")

    def set_saved_steps(self, steps: int):
        raise RuntimeError("Cannot change saved_steps on a ProxyValueBuffer")

    def _advance(self, value: np.ndarray) -> None:
        raise RuntimeError("Cannot advance a ProxyValueBuffer")

    def reset(self):
        pass
