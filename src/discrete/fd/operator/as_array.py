import numpy as np
import scipy.sparse as sp

from algebra.expression import ConstSparseExpression
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
    n = int(np.prod(shape))
    rows, cols, data = [], [], []

    for linear_idx in range(n):
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

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    expr = ConstSparseExpression(matrix)
    return ArrayOperator(operator.space, operator.shape_transform, expr)
