import math

import numpy as np
import sparse as sp

from algebra.operator import ArrayOperator
from algebra.expression import ConstSparseExpression
from algebra.space import Space, ShapeTransform
from discrete.fd.stencil import AxStencil, StencilOperator


def _axstencil_to_array(space: Space, ax: int, stencil: AxStencil) -> sp.COO:
    # Represent a single AxStencil applied on a single axis as a n x n sparse
    # array. Column j is the raveled result of applying the stencil to the j-th
    # unit basis field, so that array @ field.ravel() reproduces eval_to.
    n = math.prod(space.shape)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    field = np.zeros(space.shape, dtype=float)
    for j in range(n):
        field.flat[j] = 1.0
        out = np.zeros_like(field)
        stencil.eval_to(ax, field, out)
        field.flat[j] = 0.0

        flat = out.ravel()
        for i in np.nonzero(flat)[0]:
            rows.append(int(i))
            cols.append(j)
            data.append(float(flat[i]))

    coords = np.array([rows, cols], dtype=int)
    return sp.COO(coords, np.array(data, dtype=float), shape=(n, n))


def as_array(operator: StencilOperator) -> ArrayOperator:
    space = operator.space
    if operator.shape_transform == ShapeTransform.NONE:
        # sum weights of AxStencils from all ax into one sparse array. create
        # ArrayOperator with sparse.COO with no components.
        mats = [
            _axstencil_to_array(space, ax, stencil)
            for ax, stencil in enumerate(operator.stencils)
        ]
        mat = mats[0]
        for extra in mats[1:]:
            mat = mat + extra
        expr = ConstSparseExpression(space, mat)
        return ArrayOperator(space, operator.shape_transform, expr)

    if operator.shape_transform in (
        ShapeTransform.INCREASE_RANK,
        ShapeTransform.REDUCE_RANK,
    ):
        # transform each AxStencil from each axis into sparse array of shape
        # (ndim, n, n) so that ArrayOperator.apply uses mat[ax] on each axis
        # correctly
        mats = [
            _axstencil_to_array(space, ax, stencil)
            for ax, stencil in enumerate(operator.stencils)
        ]
        mat = sp.stack(mats, axis=0)
        expr = ConstSparseExpression(space, mat)
        return ArrayOperator(space, operator.shape_transform, expr)

    raise ValueError(f"Unknown ShapeTransform: {operator.shape_transform}")
