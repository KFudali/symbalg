from __future__ import annotations
import numpy as np

from algebra.space import Shape
from algebra.expression import FieldExpression, CallableExpression
from algebra.domain.bcs import BoundaryCondition, BoundaryTool
from algebra.operator import ArrayOperator
from algebra.domain import DomainOperator

from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        bc_tool: BoundaryTool,
        lhs: DomainOperator,
        rhs: FieldExpression,
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
        if isinstance(self._lhs, ArrayOperator):
            lhs = self._bc_tool.apply_bcs_array(self._bcs, self._lhs, rhs)
        else:
            lhs = self._lhs.apply_bcs(self._bcs, rhs)
        system = LinearSystem(lhs, rhs)
        for constraint in self._constraints:
            system = constraint.apply(system)
        return system

    def _normalize(self, result: np.ndarray):
        self._bc_tool.normalize(self._bcs, result)

    def solve(self, solver: LinearSolver) -> FieldExpression:
        def _solve() -> np.ndarray:
            system = self._assemble()
            out = solver.solve(system)
            self._normalize(out)
            return out

        result_shape = Shape.from_array(
            self._lhs.space, self._rhs_expr.shape.fieldshape()
        )
        return CallableExpression(result_shape, _solve)
