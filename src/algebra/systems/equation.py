import numpy as np

from algebra.expression import Expression, CallableExpression
from .bcs import BoundaryCondition, BoundaryTool
from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        bc_tool: BoundaryTool,
        system: LinearSystem,
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint]
    ):
        self._bc_tool = bc_tool
        self._system = system
        self._bcs = bcs
        self._constraints = constraints

    def assemble(self) -> LinearSystem:
        system = self._system.copy()
        self._bc_tool.apply(self._bcs, system)
        for constraint in self._constraints:
            constraint.apply(system)
        return system

    def solve(self, solver: LinearSolver) -> Expression:
        system = self.assemble()

        def _solve() -> np.ndarray:
            out = solver.solve(system)
            self._bc_tool.post_solve(self._bcs, out)
            return out

        return CallableExpression(system.rhs.shape, _solve)
