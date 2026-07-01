import numpy as np
from tools.buffer import ValueBuffer


class ConstValueBuffer(ValueBuffer):
    def __init__(self, array: np.ndarray, saved_steps: int = 1):
        self._array = array
        self._saved_steps = saved_steps

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def saved_steps(self) -> int:
        return self._saved_steps

    def get(self, index: int = 0) -> np.ndarray:
        return self._array

    def set(self, value: np.ndarray):
        self._array = value

    def set_saved_steps(self, steps: int):
        self._saved_steps = steps

    def advance(self, value: np.ndarray | None = None):
        pass

    def reset(self):
        pass

    def _advance(self, value: np.ndarray) -> None:
        pass
