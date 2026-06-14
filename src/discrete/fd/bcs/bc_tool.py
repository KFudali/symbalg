from typing import Callable
import numpy as np

from algebra.systems.bcs import BoundaryTool, BoundaryCondition, BCType
from algebra.systems.systems import LinearSystem
from discrete.fd.domain import FDBoundary
from discrete.fd.domain.fd_domain import FDDomain
from discrete.fd.operators import FDOperator
from discrete.fd.tools.stencil import AxStencil
from . import dirichlet, neumann

BcApplyCallable = Callable[[AxStencil, FDBoundary, float, np.ndarray], AxStencil]
BcPostSolveCallable = Callable[[FDBoundary, float, np.ndarray], None]


class FDBCTool(BoundaryTool[FDOperator]):
    APPLY: dict[BCType, BcApplyCallable] = {
        BCType.DIRICHLET: dirichlet.apply,
        BCType.NEUMANN: neumann.apply,
    }
    POST_SOLVE: dict[BCType, BcPostSolveCallable] = {
        BCType.DIRICHLET: dirichlet.post_solve,
        BCType.NEUMANN: neumann.post_solve,
    }

    def __init__(self, domain: FDDomain):
        self._domain = domain

    def apply(
        self,
        bcs: list[BoundaryCondition],
        system: LinearSystem[FDOperator],
    ) -> LinearSystem[FDOperator]:
        system = system.copy()
        lhs = system.lhs
        for bc in bcs:
            boundary = self._domain.boundary(bc.id)
            stencil = lhs.stencils[boundary.ax]
            modified_stencil = FDBCTool.APPLY[bc.bc_type](
                stencil, boundary, bc.value, system.rhs
            )
            lhs = lhs.modify(boundary.ax, modified_stencil)
        return LinearSystem(lhs, system.rhs)

    def post_solve(self, bcs: list[BoundaryCondition], field: np.ndarray) -> None:
        for bc in bcs:
            boundary = self._domain.boundary(bc.id)
            FDBCTool.POST_SOLVE[bc.bc_type](boundary, bc.value, field)
