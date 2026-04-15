from abc import ABC, abstractmethod
from algebra.systems.bcs import BoundaryCondition
from algebra.operator import Operator

class BCTool(ABC):
    @abstractmethod
    def apply(self, bc: BoundaryCondition, operator: Operator):
        pass
