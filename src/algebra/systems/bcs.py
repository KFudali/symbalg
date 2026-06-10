from abc import ABC, abstractmethod
import numpy as np
from algebra.systems.systems import LinearSystem


class BoundaryCondition(ABC):
    @abstractmethod
    def apply(self, system: LinearSystem) -> LinearSystem:
        pass

    @abstractmethod
    def post_solve(self, field: np.ndarray):
        pass
