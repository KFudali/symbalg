from typing import Callable
import numpy as np
from discrete.core.bcs import BCTool, BoundaryCondition, BCType
from discrete.fd.domain import FDBoundary
from discrete.fd.operators import FDOperator
from discrete.fd.tools.stencil import AxStencil
from . import dirichlet, neumann

BcApplyCallable = Callable[[AxStencil, BoundaryCondition, np.ndarray], AxStencil]


class FDBCTool(BCTool[FDOperator, FDBoundary]):
    APPLY: dict[BCType, BcApplyCallable] = {
        BCType.DIRICHLET: dirichlet.apply,
        BCType.NEUMANN: neumann.apply,
    }
    POST_SOLVE: dict[BCType, Callable[[BoundaryCondition, np.ndarray], None]] = {
        BCType.DIRICHLET: dirichlet.post_solve,
        BCType.NEUMANN: neumann.post_solve,
    }

    def apply(
        self, bcs: list[BoundaryCondition[FDBoundary]], lhs: FDOperator, rhs: np.ndarray
    ) -> FDOperator:
        for bc in bcs:
            stencil = lhs.stencils[bc.boundary.ax]
            modified_stencil = FDBCTool.APPLY[bc.bc_type](stencil, bc, rhs)
            lhs = lhs.modify(bc.boundary.ax, modified_stencil)
        return lhs

    def post_solve(self, bc: BoundaryCondition, field: np.ndarray):
        FDBCTool.POST_SOLVE[bc.bc_type](bc, field)
