from abc import ABC, abstractmethod
from typing import Generic
import numpy as np

from algebra.systems.systems import LinearSystem
from algebra.operator import TOperator
from .boundary_condition import BoundaryCondition


class BoundaryTool(ABC, Generic[TOperator]):
    @abstractmethod
    def apply(
        self, bcs: list[BoundaryCondition], system: LinearSystem[TOperator]
    ) -> LinearSystem[TOperator]:
        pass

    @abstractmethod
    def post_solve(self, bcs: list[BoundaryCondition], field: np.ndarray):
        pass
