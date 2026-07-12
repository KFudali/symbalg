from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from algebra.systems.systems import LinearSystem
from algebra.operator import OperatorWrapper
from algebra.domain.bcs import BoundaryCondition


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


@dataclass(frozen=True)
class SystemConstraints:
    constraints: list[SystemConstraint] = field(default_factory=[])
    bcs: list[BoundaryCondition] = field(default_factory=[])
