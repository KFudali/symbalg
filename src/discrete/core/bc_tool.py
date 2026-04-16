from abc import ABC, abstractmethod
from typing import Self
from algebra.systems.bcs import BoundaryCondition
from algebra.symbolic.affine_operator import AffineOperator

class BCTool(ABC):
    @abstractmethod
    def apply(
        self,
        bcs: list[BoundaryCondition],
        operator: AffineOperator
    ) -> AffineOperator:
        pass
