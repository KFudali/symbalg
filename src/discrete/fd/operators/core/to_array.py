from functools import reduce

import scipy.sparse as sp

from discrete.fd.tools.stencil import AxStencil

from algebra.space import Space, ShapeTransform


def _axis_matrix(ax_stencil: AxStencil, n: int) -> sp.csr_matrix:
    rows, cols, data = [], [], []

    n_left = len(ax_stencil.lefts)
    n_right = len(ax_stencil.rights)

    for i in range(n_left):
        stencil = ax_stencil.lefts[i]
        for offset, weight in stencil.weights.items():
            c = i + offset
            if 0 <= c < n:
                rows.append(i)
                cols.append(c)
                data.append(weight)

    for j in range(n_right):
        i = n - 1 - j
        stencil = ax_stencil.rights[j]
        for offset, weight in stencil.weights.items():
            c = i + offset
            if 0 <= c < n:
                rows.append(i)
                cols.append(c)
                data.append(weight)

    for i in range(n_left, n - n_right):
        for offset, weight in ax_stencil.interior.weights.items():
            c = i + offset
            if 0 <= c < n:
                rows.append(i)
                cols.append(c)
                data.append(weight)

    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _extend_to_nd(
    axis_1d: list[sp.csr_matrix], shape: tuple[int, ...]
) -> list[sp.csr_matrix]:
    eye_list = [sp.eye(n, format="csr") for n in shape]
    ndim = len(shape)
    per_axis_nd = []
    for ax in range(ndim):
        mats = list(eye_list)
        mats[ax] = axis_1d[ax]
        per_axis_nd.append(reduce(sp.kron, mats))
    return per_axis_nd


def _combine_axes(
    per_axis_nd: list[sp.csr_matrix], transform: ShapeTransform
) -> sp.csr_matrix:
    if transform == ShapeTransform.NONE:
        result = sum(per_axis_nd[1:], per_axis_nd[0])
    elif transform == ShapeTransform.INCREASE_RANK:
        result = sp.vstack(per_axis_nd)
    elif transform == ShapeTransform.REDUCE_RANK:
        result = sp.hstack(per_axis_nd)
    else:
        raise ValueError(f"Unknown ShapeTransform: {transform}")
    return result.tocsr()


def to_array(
    stencils: tuple[AxStencil, ...], space: Space, shape_transform: ShapeTransform
) -> sp.spmatrix:
    axis_1d = [
        _axis_matrix(stencil, n) for stencil, n in zip(stencils, space.shape)
    ]
    per_axis_nd = _extend_to_nd(axis_1d, space.shape)
    return _combine_axes(per_axis_nd, shape_transform)
