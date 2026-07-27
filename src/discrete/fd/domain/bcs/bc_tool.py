from typing import Callable
import numpy as np
import scipy.sparse as sp

from algebra.space import Space
from algebra.expression import ConstSparseExpression
from algebra.operator import ArrayOperator
from algebra.domain import BoundaryId
from algebra.domain.bcs import BoundaryTool, BoundaryCondition, BCType, BCValue
from discrete.fd.domain import FDBoundary
from discrete.fd.stencil import AxStencil, StencilOperator

from . import dirichlet, neumann

BcApplyCallable = Callable[[AxStencil, FDBoundary, float, np.ndarray], AxStencil]
BcApplyArrayCallable = Callable[
    [sp.spmatrix, FDBoundary, float, np.ndarray], sp.spmatrix
]
BcPostSolveCallable = Callable[[FDBoundary, float, np.ndarray], None]


def _component_value(value: BCValue, comp: int) -> BCValue:
    """Slice the leading axis of a per-component BC value, or broadcast scalars."""
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[comp]
    return value


class FDBCTool(BoundaryTool[StencilOperator]):
    APPLY: dict[BCType, BcApplyCallable] = {
        BCType.DIRICHLET: dirichlet.apply,
        BCType.NEUMANN: neumann.apply,
    }
    APPLY_ARRAY: dict[BCType, BcApplyArrayCallable] = {
        BCType.DIRICHLET: dirichlet.apply_array,
        BCType.NEUMANN: neumann.apply_array,
    }
    POST_SOLVE: dict[BCType, BcPostSolveCallable] = {
        BCType.DIRICHLET: dirichlet.post_solve,
        BCType.NEUMANN: neumann.post_solve,
    }

    def __init__(self, space: Space, boundaries: dict[BoundaryId, FDBoundary]):
        self._space = space
        self._boundaries = boundaries

    @property
    def space(self) -> Space:
        return self._space

    def apply_bcs_array(
        self, bcs: list[BoundaryCondition], lhs: ArrayOperator, rhs: np.ndarray
    ) -> ArrayOperator:
        mat = lhs.mat.eval()
        for bc in bcs:
            boundary = self._boundaries[bc.boundary]
            mat = self._apply_rankwise_array(
                FDBCTool.APPLY_ARRAY[bc.bc_type],
                mat,
                boundary,
                bc.value,
                rhs,
            )
        return ArrayOperator(
            lhs.space, lhs.shape_transform, ConstSparseExpression(lhs.space, mat)
        )

    def apply_bcs(
        self, bcs: list[BoundaryCondition], lhs: StencilOperator, rhs: np.ndarray
    ) -> StencilOperator:
        for bc in bcs:
            boundary = self._boundaries[bc.boundary]
            stencil = lhs.stencils[boundary.ax]
            modified_stencil = self._apply_rankwise(
                FDBCTool.APPLY[bc.bc_type],
                stencil,
                boundary,
                bc.value,
                rhs,
            )
            lhs = lhs.modify(boundary.ax, modified_stencil)
        return lhs

    def normalize(self, bcs: list[BoundaryCondition], field: np.ndarray) -> None:
        for bc in bcs:
            boundary = self._boundaries[bc.boundary]
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
        value: BCValue,
        rhs: np.ndarray,
    ) -> AxStencil:
        if rhs.ndim == self.space.ndim:
            return fn(stencil, boundary, float(value), rhs)
        modified = stencil
        for comp in range(rhs.shape[0]):
            modified = self._apply_rankwise(
                fn, stencil, boundary, _component_value(value, comp), rhs[comp]
            )
        return modified

    def _apply_rankwise_array(
        self,
        fn: BcApplyArrayCallable,
        mat: sp.spmatrix,
        boundary: FDBoundary,
        value: BCValue,
        rhs: np.ndarray,
    ) -> sp.spmatrix:
        if rhs.ndim == self.space.ndim:
            return fn(mat, boundary, float(value), rhs)
        modified = mat
        for comp in range(rhs.shape[0]):
            modified = self._apply_rankwise_array(
                fn, modified, boundary, _component_value(value, comp), rhs[comp]
            )
        return modified

    def _post_solve_rankwise(
        self,
        fn: BcPostSolveCallable,
        boundary: FDBoundary,
        value: BCValue,
        field: np.ndarray,
    ) -> None:
        if field.ndim == self._space.ndim:
            fn(boundary, float(value), field)
            return
        for comp in range(field.shape[0]):
            self._post_solve_rankwise(
                fn, boundary, _component_value(value, comp), field[comp]
            )
