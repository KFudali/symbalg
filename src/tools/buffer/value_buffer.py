from abc import ABC, abstractmethod
import numpy as np


class ValueBuffer(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def saved_steps(self) -> int:
        pass

    @abstractmethod
    def get(self, index: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def set(self, value: np.ndarray):
        pass

    @abstractmethod
    def set_saved_steps(self, steps: int):
        pass

    def advance(self, value: np.ndarray | None = None):
        if value is None:
            # Default: carry the current value forward so that, after advance,
            # `value()` still reflects the most recently written state and
            # `past(1)` becomes what was previously current. This gives a
            # warm-start semantics for time-stepping loops.
            value = self.get(0).copy()
        self._advance(value)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def _advance(self, value: np.ndarray) -> None:
        pass
