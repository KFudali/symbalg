import discrete.core as discr

from algebra.systems.bcs import BoundaryCondition, BCType
from algebra.expression import Expression

from algebra.operator import Operator
from algebra.systems import LinearEquation, SystemConstraint


class BCFactory:
    def __init__(self, domain: discr.domain.Domain):
        self._domain = domain

    def dirichlet(
        self, boundary_id: discr.domain.BoundaryId, value: float
    ) -> BoundaryCondition:
        return BoundaryCondition(BCType.DIRICHLET, value, boundary_id)

    def neumann(
        self, boundary_id: discr.domain.BoundaryId, value: float
    ) -> BoundaryCondition:
        return BoundaryCondition(BCType.NEUMANN, value, boundary_id)


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
        constraints: list[SystemConstraint]
    ) -> LinearEquation:
        return LinearEquation(self._bc_tool, lhs, rhs, bcs, constraints=constraints)
