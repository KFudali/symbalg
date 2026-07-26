from abc import ABC, abstractmethod
from typing import Generic
import numpy as np

from algebra.space import Space
from algebra.operator import TOperator, ArrayOperator
from .bcs import BoundaryCondition


class BoundaryTool(ABC, Generic[TOperator]):
    @property
    @abstractmethod
    def space(self) -> Space: ...

    @abstractmethod
    def apply_bcs(
        self, bcs: list[BoundaryCondition], lhs: TOperator, rhs: np.ndarray
    ) -> TOperator: ...

    @abstractmethod
    def apply_bcs_array(
        self, bcs: list[BoundaryCondition], lhs: ArrayOperator, rhs: np.ndarray
    ) -> ArrayOperator: ...

    @abstractmethod
    def normalize(self, bcs: list[BoundaryCondition], field: np.ndarray): ...
