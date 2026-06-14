import numpy as np

from algebra.operator import Operator
from algebra.expression import Expression, CallableExpression
from .bcs import BoundaryCondition, BoundaryTool
from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        bc_tool: BoundaryTool,
        lhs: Operator,
        rhs: Expression,
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint],
    ):
        self._bc_tool = bc_tool
        self._lhs = lhs
        self._rhs = rhs
        self._bcs = bcs
        self._constraints = constraints

    def _assemble(self) -> LinearSystem:
        system = LinearSystem(self._lhs.copy(), self._rhs.eval().copy())
        self._bc_tool.apply(self._bcs, system)
        for constraint in self._constraints:
            constraint.apply(system)
        return system

    def solve(self, solver: LinearSolver) -> Expression:
        def _solve() -> np.ndarray:
            system = self._assemble()
            out = solver.solve(system)
            self._bc_tool.post_solve(self._bcs, out)
            return out

        return CallableExpression(self._rhs.shape, _solve)
