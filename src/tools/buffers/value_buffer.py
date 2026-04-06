from abc import ABC, abstractmethod
import numpy as np


class ValueBuffer(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]: pass

    @property
    @abstractmethod
    def saved_steps(self) -> int: pass

    @abstractmethod
    def get(self, index: int) -> np.ndarray: pass
    
    @abstractmethod
    def set(self, value: np.ndarray): pass

    @abstractmethod
    def set_saved_steps(self, steps: int): pass

    def advance(self, value: np.ndarray | None = None):
        if value is None:
            value = np.zeros(self.shape)
        self._advance(value)

    @abstractmethod
    def _advance(self, value: np.ndarray) -> None: pass



