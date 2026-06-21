from typing import Callable, Union
import numpy as np

from algebra.systems.bcs import BoundaryTool, BoundaryCondition, BCType
from algebra.systems.systems import LinearSystem
from discrete.fd.domain import FDBoundary
from discrete.fd.domain.fd_domain import FDDomain
from discrete.fd.operators import FDOperator
from discrete.fd.tools.stencil import AxStencil
from . import dirichlet, neumann

BCValueLike = Union[float, np.ndarray]
BcApplyCallable = Callable[[AxStencil, FDBoundary, float, np.ndarray], AxStencil]
BcPostSolveCallable = Callable[[FDBoundary, float, np.ndarray], None]


def _component_value(value: BCValueLike, comp: int) -> BCValueLike:
    """Slice the leading axis of a per-component BC value, or broadcast scalars."""
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[comp]
    return value


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
            modified_stencil = self._apply_rankwise(
                FDBCTool.APPLY[bc.bc_type],
                stencil,
                boundary,
                bc.value,
                system.rhs,
            )
            lhs = lhs.modify(boundary.ax, modified_stencil)
        return LinearSystem(lhs, system.rhs)

    def post_solve(self, bcs: list[BoundaryCondition], field: np.ndarray) -> None:
        for bc in bcs:
            boundary = self._domain.boundary(bc.id)
            self._post_solve_rankwise(
                FDBCTool.POST_SOLVE[bc.bc_type],
                boundary,
                bc.value,
                field,
            )

    def _apply_rankwise(
        self,
        fn: BcApplyCallable,
        stencil: AxStencil,
        boundary: FDBoundary,
        value: BCValueLike,
        rhs: np.ndarray,
    ) -> AxStencil:
        if rhs.ndim == self._domain.grid.ndim:
            return fn(stencil, boundary, float(value), rhs)
        modified = stencil
        for comp in range(rhs.shape[0]):
            modified = self._apply_rankwise(
                fn, stencil, boundary, _component_value(value, comp), rhs[comp]
            )
        return modified

    def _post_solve_rankwise(
        self,
        fn: BcPostSolveCallable,
        boundary: FDBoundary,
        value: BCValueLike,
        field: np.ndarray,
    ) -> None:
        if field.ndim == self._domain.grid.ndim:
            fn(boundary, float(value), field)
            return
        for comp in range(field.shape[0]):
            self._post_solve_rankwise(
                fn, boundary, _component_value(value, comp), field[comp]
            )
