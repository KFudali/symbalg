from __future__ import annotations
import numpy as np

from algebra.space import FieldShape
from algebra.expression import Expression, CallableExpression
from algebra.domain.bcs import BoundaryCondition, BoundaryTool
from algebra.domain import DomainOperator

from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        bc_tool: BoundaryTool,
        lhs: DomainOperator,
        rhs: Expression,
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint],
    ):
        self._bc_tool = bc_tool
        self._lhs = lhs
        self._rhs_expr = rhs
        self._bcs = bcs
        self._constraints = constraints

    def _assemble(self) -> LinearSystem:
        rhs = self._rhs_expr.eval()
        lhs = self._lhs.apply_bcs(self._bcs, rhs)
        system = LinearSystem(lhs, rhs)
        for constraint in self._constraints:
            system = constraint.apply(system)
        return system

    def _normalize(self, result: np.ndarray):
        self._bc_tool.normalize(self._bcs, result)

    def solve(self, solver: LinearSolver) -> Expression:
        def _solve() -> np.ndarray:
            system = self._assemble()
            out = solver.solve(system)
            self._normalize(out)
            return out

        result_shape = FieldShape.from_shape(
            self._lhs.space, self._rhs_expr.shape
        )
        return CallableExpression(result_shape, _solve)
