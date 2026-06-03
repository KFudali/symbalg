import numpy as np
from tools import region

from discrete.core.bcs import BoundaryCondition
from discrete.fd.tools.stencil import AxStencil, Stencil
from discrete.fd.domain import FDBoundary


def apply(
    stencil: AxStencil,
    bc: BoundaryCondition[FDBoundary],
    rhs: np.ndarray,
) -> AxStencil:
    _add_boundary_rhs_contribution(stencil, bc, rhs)
    return _stencil(bc.boundary.side, stencil)


def _add_boundary_rhs_contribution(
    stencil: AxStencil, bc: BoundaryCondition[FDBoundary], rhs: np.ndarray
):
    field = np.zeros_like(rhs)
    b = bc.boundary
    boundary = region.boundary(field.ndim, b.ax, b.side, b.exclude_corners)
    field[boundary] = -bc.value
    for ax in range(rhs.ndim):
        stencil.eval_to(ax, field, rhs)
    rhs[boundary] = 0.0


def _stencil(side: int, stencil: AxStencil) -> AxStencil:
    rights = list(stencil.rights)
    lefts = list(stencil.lefts)
    if side == -1:
        lefts[0] = Stencil({0: 1.0})
    else:
        rights[0] = Stencil({0: 1.0})
    return AxStencil(stencil.interior, tuple(lefts), tuple(rights))


def post_solve(bc: BoundaryCondition, field: np.ndarray):
    boundary = region.boundary(field.ndim, bc.boundary.ax, bc.boundary.side)
    field[boundary] = bc.value
