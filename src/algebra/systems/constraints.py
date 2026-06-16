from abc import ABC, abstractmethod
import numpy as np

from algebra.systems.systems import LinearSystem
from algebra.operator_wrapper import OperatorWrapper


class SystemConstraint(ABC):
    @abstractmethod
    def apply(self, system: LinearSystem) -> LinearSystem:
        pass


class FixedMeanConstraint(SystemConstraint):
    def apply(self, system: LinearSystem) -> LinearSystem:
        def _force_fixed_mean(inp: np.ndarray, out: np.ndarray):
            out[:] -= out.mean()
        mean_wrapper = OperatorWrapper(system.lhs, _force_fixed_mean)
        rhs = system.rhs - system.rhs.mean()
        return LinearSystem(mean_wrapper, rhs)
