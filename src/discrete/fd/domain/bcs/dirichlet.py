import numpy as np
import scipy.sparse as sp

from tools import region

from discrete.fd.stencil import AxStencil, Stencil
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


def apply_array(
    mat: sp.spmatrix,
    boundary: FDBoundary,
    value: float,
    rhs: np.ndarray,
) -> sp.spmatrix:
    """Apply a Dirichlet BC directly to a sparse matrix + rhs.

    Mirrors the stencil-form ``apply``: moves the known boundary contribution
    to the rhs, then replaces the boundary rows with an identity row and sets
    ``rhs`` at the boundary indices to ``value``.
    """
    shape = rhs.shape
    b_indices = _boundary_linear_indices(shape, boundary)
    rhs_flat = rhs.reshape(-1)

    csr = sp.csr_matrix(mat)

    # Move known boundary column contributions to rhs.
    boundary_cols = csr[:, b_indices]
    rhs_flat -= np.asarray(boundary_cols.sum(axis=1)).ravel() * value

    # Zero rows and columns at boundary indices, set identity on diagonal.
    lil = sp.lil_matrix(csr)
    lil[b_indices, :] = 0.0
    lil[:, b_indices] = 0.0
    for i in b_indices:
        lil[i, i] = 1.0

    rhs_flat[b_indices] = value
    return lil.tocsr()


def _boundary_linear_indices(
    shape: tuple[int, ...], boundary: FDBoundary
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    b_region = region.boundary(
        len(shape), boundary.ax, boundary.side, boundary.exclude_corners
    )
    mask[b_region] = True
    return np.flatnonzero(mask.ravel())
