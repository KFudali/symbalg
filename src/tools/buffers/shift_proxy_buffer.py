import numpy as np
from .value_buffer import ValueBuffer


class ShiftProxyValueBuffer(ValueBuffer):
    def __init__(self, base: ValueBuffer, shift: int):
        if shift < 0:
            raise ValueError("shift must be >= 0")

        if shift >= base.saved_steps:
            raise ValueError(
                f"Cannot create past proxy with shift={shift}; "
                f"base buffer only has {base.saved_steps} saved steps"
            )

        self._base = base
        self._shift = shift

    @property
    def shape(self) -> tuple[int, ...]:
        return self._base.shape

    @property
    def saved_steps(self) -> int:
        return self._base.saved_steps - self._shift

    def get(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError("index must be >= 0")
        base_index = index + self._shift
        if base_index >= self._base.saved_steps:
            raise IndexError(
                f"Requested index {index} exceeds available past history"
            )
        return self._base.get(base_index)
    
    def set(self, value: np.ndarray):
        raise RuntimeError(
            "Cannot set value on a PastProxyValueBuffer (read-only view)"
        )

    def set_saved_steps(self, steps: int):
        raise RuntimeError(
            "Cannot change saved_steps on a PastProxyValueBuffer"
        )

    def _advance(self, value: np.ndarray):
        raise RuntimeError(
            "Cannot advance a PastProxyValueBuffer"
        )