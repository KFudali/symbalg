import numpy as np
from tools import region

from discrete.fd.tools.stencil import AxStencil, Stencil
from discrete.fd.domain import FDBoundary


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
    field = np.zeros_like(rhs)
    b_region = region.boundary(
        field.ndim, boundary.ax, boundary.side, boundary.exclude_corners
    )
    field[b_region] = -value
    for ax in range(rhs.ndim):
        stencil.eval_to(ax, field, rhs)
    rhs[b_region] = 0.0


def _stencil(side: int, stencil: AxStencil) -> AxStencil:
    rights = list(stencil.rights)
    lefts = list(stencil.lefts)
    if side == -1:
        lefts[0] = Stencil({0: 1.0})
    else:
        rights[0] = Stencil({0: 1.0})
    return AxStencil(stencil.interior, tuple(lefts), tuple(rights))


def post_solve(boundary: FDBoundary, value: float, field: np.ndarray):
    b_region = region.boundary(field.ndim, boundary.ax, boundary.side)
    field[b_region] = value
