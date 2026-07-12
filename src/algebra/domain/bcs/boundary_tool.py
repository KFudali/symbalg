from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic
import numpy as np

from algebra.operator import TOperator
from .boundary_condition import BoundaryCondition

if TYPE_CHECKING:
    from algebra.systems.systems import LinearSystem


class BoundaryTool(ABC, Generic[TOperator]):
    @abstractmethod
    def apply(
        self, bcs: list[BoundaryCondition], system: LinearSystem[TOperator]
    ) -> LinearSystem[TOperator]:
        pass

    @abstractmethod
    def post_solve(self, bcs: list[BoundaryCondition], field: np.ndarray):
        pass
