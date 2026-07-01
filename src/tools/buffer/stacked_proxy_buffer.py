import numpy as np
from .value_buffer import ValueBuffer


class StackedProxyValueBuffer(ValueBuffer):
    def __init__(self, sources: tuple[ValueBuffer, ...], ax: int = 0):
        try:
            shape = np.stack(tuple(source.get() for source in sources), axis=ax).shape
        except Exception as e:
            shapes = tuple(source.shape for source in sources)
            raise ValueError(
                f"Cannot stack buffers of shapes: {shapes} over ax: {ax}."
            ) from e
        self._sources = sources
        self._shape = shape
        self._ax = ax

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def saved_steps(self) -> int:
        min_step = min(source.saved_steps for source in self._sources)
        return min_step

    def get(self, index: int = 0) -> np.ndarray:
        return np.stack(
            tuple(source.get(index) for source in self._sources), axis=self._ax
        )

    def set(self, value: np.ndarray):
        raise RuntimeError("Cannot set value on a ProxyValueBuffer (read-only view)")

    def set_saved_steps(self, steps: int):
        raise RuntimeError("Cannot change saved_steps on a ProxyValueBuffer")

    def _advance(self, value: np.ndarray) -> None:
        raise RuntimeError("Cannot advance a ProxyValueBuffer")

    def reset(self):
        pass
