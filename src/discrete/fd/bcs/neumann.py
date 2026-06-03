import numpy as np

from discrete.core.bcs import BoundaryCondition
from discrete.fd.tools.stencil import AxStencil, Stencil
from discrete.fd.domain import FDBoundary
from tools import region


def apply(
    stencil: AxStencil, bc: BoundaryCondition[FDBoundary], rhs: np.ndarray
) -> AxStencil:
    _add_boundary_rhs_contribution(stencil, bc, rhs)
    return _stencil(bc.boundary.side, stencil)


def _add_boundary_rhs_contribution(
    stencil: AxStencil, bc: BoundaryCondition[FDBoundary], rhs: np.ndarray
):
    b = bc.boundary
    boundary = region.boundary(rhs.ndim, b.ax, b.side, b.exclude_corners)
    left, right = stencil.interior.max_offsets()
    if b.side == -1:
        rhs[boundary] -= 2.0 * bc.value * b.dh * stencil.interior[left]
    else:
        rhs[boundary] -= 2.0 * bc.value * b.dh * stencil.interior[right]


def _stencil(side: int, stencil: AxStencil) -> AxStencil:
    rights = list(stencil.rights)
    lefts = list(stencil.lefts)
    left, right = stencil.interior.max_offsets()
    if left != 1 or right != 1:
        raise ValueError("Neumman bc only handles single ghost node")
    if side == -1:
        ghost_weight = stencil.interior.weights[-1]
        weight = stencil.interior.weights.get(1, 0.0)
        st = Stencil(
            {0: stencil.interior.weights.get(0, 0.0), 1: weight + ghost_weight}
        )
        lefts[0] = st
    else:
        ghost_weight = stencil.interior.weights[1]
        weight = stencil.interior.weights.get(-1, 0.0)
        st = Stencil(
            {0: stencil.interior.weights.get(0, 0.0), -1: weight + ghost_weight}
        )
        rights[0] = st
    return AxStencil(stencil.interior, tuple(lefts), tuple(rights))


def post_solve(bc: BoundaryCondition[FDBoundary], field: np.ndarray):
    pass
