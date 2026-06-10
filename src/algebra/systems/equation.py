import numpy as np

from algebra.expression import Expression, CallableExpression
from .bcs import BoundaryCondition
from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        system: LinearSystem,
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint]
    ):
        self._system = system
        self._bcs = bcs
        self._constraints = constraints

    def assemble(self) -> LinearSystem:
        system = self._system.copy()
        for bc in self._bcs:
            bc.apply(system)
        for constraint in self._constraints:
            constraint.apply(system)
        return system

    def solve(self, solver: LinearSolver) -> Expression:
        system = self.assemble()

        def _solve() -> np.ndarray:
            return solver.solve(system)

        return CallableExpression(system.rhs.shape, _solve)
