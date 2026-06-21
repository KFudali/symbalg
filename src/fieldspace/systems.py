from typing import Sequence, Union

import numpy as np

import discrete.core as discr

from algebra.systems.bcs import BoundaryCondition, BCType
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
    def __init__(self, domain: discr.domain.Domain):
        self._domain = domain

    def dirichlet(
        self, boundary_id: discr.domain.BoundaryId, value: BCValueInput
    ) -> BoundaryCondition:
        return BoundaryCondition(
            BCType.DIRICHLET, _normalize_bc_value(value), boundary_id
        )

    def neumann(
        self, boundary_id: discr.domain.BoundaryId, value: BCValueInput
    ) -> BoundaryCondition:
        return BoundaryCondition(
            BCType.NEUMANN, _normalize_bc_value(value), boundary_id
        )


class SystemFactory:
    def __init__(self, discrete: discr.Discretization):
        self._bc_tool = discrete.bc_tool
        self._bc_factory = BCFactory(discrete.domain)

    @property
    def bc(self) -> BCFactory:
        return self._bc_factory

    def les(
        self,
        lhs: Operator,
        rhs: Expression,
        bcs: list[BoundaryCondition],
        *,
        constraints: list[SystemConstraint] = []
    ) -> LinearEquation:
        if isinstance(lhs, AffineOperator):
            rhs -= lhs.expression
            lhs = lhs.operator
        return LinearEquation(self._bc_tool, lhs, rhs, bcs, constraints=constraints)
