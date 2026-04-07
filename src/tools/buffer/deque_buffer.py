from collections import deque
import numpy as np

from .value_buffer import ValueBuffer


class DequeValueBuffer(ValueBuffer):
    def __init__(self, shape: tuple[int, ...], saved_steps: int = 1):
        if saved_steps < 1:
            raise ValueError("saved_steps must be >= 1")

        self._shape = shape
        self._saved_steps = saved_steps

        self._buffer = deque(
            (np.zeros(shape=shape) for _ in range(saved_steps)), maxlen=saved_steps
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def saved_steps(self) -> int:
        return self._saved_steps

    def get(self, index: int) -> np.ndarray:
        if index < 0 or index >= self._saved_steps:
            raise IndexError(
                f"Requested step {index}, "
                f"but buffer stores {self._saved_steps} steps"
            )
        return self._buffer[index]

    def set(self, value: np.ndarray):
        value = np.asarray(value)
        if value.shape != self._shape:
            raise ValueError(
                f"Shape mismatch: expected {self._shape}, got {value.shape}"
            )
        self._buffer[0][...] = value

    def set_saved_steps(self, steps: int):
        if steps < 1:
            raise ValueError("saved_steps must be >= 1")

        if steps == self._saved_steps:
            return

        new_buffer = deque(maxlen=steps)

        for i in range(min(steps, self._saved_steps)):
            new_buffer.append(self._buffer[i].copy())

        while len(new_buffer) < steps:
            new_buffer.append(np.zeros(self._shape))

        self._buffer = new_buffer
        self._saved_steps = steps

    def _advance(self, value: np.ndarray) -> None:
        value = np.asarray(value)
        if value.shape != self._shape:
            raise ValueError(
                f"Shape mismatch: expected {self._shape}, got {value.shape}"
            )

        self._buffer.appendleft(value.copy())
