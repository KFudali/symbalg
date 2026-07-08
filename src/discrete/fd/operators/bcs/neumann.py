import numpy as np

from discrete.fd.tools.stencil import AxStencil, Stencil
from discrete.fd.domain import FDBoundary
from tools import region


def apply(
    stencil: AxStencil,
    boundary: FDBoundary,
    value: float,
    rhs: np.ndarray,
) -> AxStencil:
    _add_boundary_rhs_contribution(stencil, boundary, value, rhs)
    return _stencil(boundary.side, stencil)


def _add_boundary_rhs_contribution(
    stencil: AxStencil, boundary: FDBoundary, value: float, rhs: np.ndarray
):
    b_region = region.boundary(
        rhs.ndim, boundary.ax, boundary.side, boundary.exclude_corners
    )
    left, right = stencil.interior.max_offsets()
    if boundary.side == -1:
        rhs[b_region] -= 2.0 * value * boundary.dh * stencil.interior[left]
    else:
        rhs[b_region] -= 2.0 * value * boundary.dh * stencil.interior[right]


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


def post_solve(boundary: FDBoundary, value: float, field: np.ndarray):
    pass
