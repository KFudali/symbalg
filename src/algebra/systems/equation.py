from __future__ import annotations
import numpy as np

from algebra.space import FieldShape
from algebra.expression import Expression, CallableExpression
from algebra.domain.bcs import BoundaryCondition
from algebra.domain import DomainOperator

from .solvers import LinearSolver
from .constraints import SystemConstraint
from .systems import LinearSystem


class LinearEquation:
    def __init__(
        self,
        system: LinearSystem[DomainOperator],
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint],
    ):
        self._system = system
        self._bcs = bcs
        self._constraints = constraints

    def _assemble(self) -> LinearSystem:
        rhs = self._system.rhs
        lhs = self._system.lhs.apply_bcs(self._bcs, rhs)
        system = LinearSystem(lhs, rhs)
        for constraint in self._constraints:
            system = constraint.apply(system)
        return system

    def _normalize(self, result: np.ndarray):
        for bc in self._bcs:
            self._system.lhs.domain.normalize_bc(bc, result)

    def solve(self, solver: LinearSolver) -> Expression:
        def _solve() -> np.ndarray:
            system = self._assemble()
            out = solver.solve(system)
            self._normalize(out)
            return out

        result_shape = FieldShape.from_shape(
            self._system.lhs.space, self._system.rhs.shape
        )
        return CallableExpression(result_shape, _solve)
