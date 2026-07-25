import numpy as np
import scipy.sparse as sp

from algebra.operator import ArrayOperator
from discrete.fd.stencil import StencilOperator, AxStencil


def _get_stencil(ax_stencil: AxStencil, pos: int, size: int):
    if pos < len(ax_stencil.lefts):
        return ax_stencil.lefts[pos]
    if pos >= size - len(ax_stencil.rights):
        return ax_stencil.rights[size - 1 - pos]
    return ax_stencil.interior


def as_array(operator: StencilOperator) -> ArrayOperator:
    shape = operator.space.shape
    n_points = int(np.prod(shape))
    rows, cols, data = [], [], []

    for linear_idx in range(n_points):
        idx = np.unravel_index(linear_idx, shape)
        for ax, size in enumerate(shape):
            stencil = _get_stencil(operator.stencils[ax], idx[ax], size)
            for offset, weight in stencil.weights.items():
                neighbor = list(idx)
                neighbor[ax] += offset
                if 0 <= neighbor[ax] < shape[ax]:
                    rows.append(linear_idx)
                    cols.append(np.ravel_multi_index(neighbor, shape))
                    data.append(weight)

    matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_points, n_points), dtype=float
    )
    return ArrayOperator(operator.space, operator.shape_transform, matrix)
