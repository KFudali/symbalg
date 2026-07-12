from typing import Sequence, Union

import numpy as np

from algebra.domain import Domain, bcs
from discrete.core import Discretization
from algebra.expression import Expression

from algebra.operator import Operator
from algebra.symbolic import AffineOperator
from algebra.systems import LinearEquation, SystemConstraint

BCValueInput = Union[float, Sequence[float], np.ndarray]


def _normalize_bc_value(value: BCValueInput) -> Union[float, np.ndarray]:
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


class BCFactory:
    def __init__(self, domain: Domain):
        self._domain = domain

    def dirichlet(
        self, boundary_id: bcs.BoundaryId, value: BCValueInput
    ) -> bcs.BoundaryCondition:
        return bcs.BoundaryCondition(
            bcs.BCType.DIRICHLET, _normalize_bc_value(value),
            self._domain.boundary(boundary_id),
        )

    def neumann(
        self, boundary_id: bcs.BoundaryId, value: BCValueInput
    ) -> bcs.BoundaryCondition:
        return bcs.BoundaryCondition(
            bcs.BCType.NEUMANN, _normalize_bc_value(value),
            self._domain.boundary(boundary_id),
        )


class SystemFactory:
    def __init__(self, discrete: Discretization):
        self._bc_tool = discrete.bc_tool
        self._bc_factory = BCFactory(discrete.domain)

    @property
    def bc(self) -> BCFactory:
        return self._bc_factory

    def les(
        self,
        lhs: Operator,
        rhs: Expression,
        bcs: list[bcs.BoundaryCondition],
        *,
        constraints: list[SystemConstraint] = []
    ) -> LinearEquation:
        if isinstance(lhs, AffineOperator):
            rhs -= lhs.expression
            lhs = lhs.operator
        return LinearEquation(self._bc_tool, lhs, rhs, bcs, constraints=constraints)
