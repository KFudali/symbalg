from abc import ABC, abstractmethod
import numpy as np
from .fieldshaped import FieldShaped

class AbstractField(FieldShaped, ABC):
    @abstractmethod
    def set_current(self, value: np.ndarray):
        pass

    @abstractmethod
    def get_current(self) -> np.ndarray:
        pass
