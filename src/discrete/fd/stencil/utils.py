import math

import numpy as np
import sparse as sp

from algebra.space import Space
from .ax_stencil import AxStencil


def stencil_to_array(space: Space, stencil: AxStencil) -> sp.SparseArray:
    # Represent AxStencil as a sparse.COO array of shape n x n where
    # n = math.prod(space.shape). The stencil operation is linear, so column j
    # of the matrix is the raveled result of applying the stencil (summed over
    # all axes) to the j-th unit basis field. This guarantees that
    # array @ field.ravel() reproduces the accumulated stencil.eval_to result.
    n = math.prod(space.shape)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    field = np.zeros(space.shape, dtype=float)
    for j in range(n):
        field.flat[j] = 1.0
        out = np.zeros_like(field)
        for ax in range(space.ndim):
            stencil.eval_to(ax, field, out)
        field.flat[j] = 0.0

        flat = out.ravel()
        nonzero = np.nonzero(flat)[0]
        for i in nonzero:
            rows.append(int(i))
            cols.append(j)
            data.append(float(flat[i]))

    coords = np.array([rows, cols], dtype=int)
    return sp.COO(coords, np.array(data, dtype=float), shape=(n, n))
