import discrete.core as discr


from algebra.expression import Expression
from algebra.operator import Operator
from .les import LES


class BCFactory:
    def __init__(self, domain: discr.domain.Domain):
        self._domain = domain

    def dirichlet(
        self, boundary_id: discr.domain.BoundaryId, value: float
    ) -> discr.bcs.BoundaryCondition:
        return discr.bcs.BoundaryCondition(
            discr.bcs.BCType.DIRICHLET, value, self._domain.boundary(boundary_id)
        )

    def neumann(
        self, boundary_id: discr.domain.BoundaryId, value: float
    ) -> discr.bcs.BoundaryCondition:
        return discr.bcs.BoundaryCondition(
            discr.bcs.BCType.NEUMANN, value, self._domain.boundary(boundary_id)
        )


class SystemFactory:
    def __init__(self, discrete: discr.Discretization):
        self._bc_tool = discrete.bc_tool
        self._bc_factory = BCFactory(discrete.domain)

    @property
    def bc(self) -> BCFactory:
        return self._bc_factory

    def les(
        self, lhs: Operator, rhs: Expression, bcs: list[discr.bcs.BoundaryCondition]
    ) -> LES:
        return LES(self._bc_tool, lhs, rhs, bcs)
