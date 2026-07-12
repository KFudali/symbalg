from __future__ import annotations

from functools import reduce
from typing import Self

import numpy as np
import scipy.sparse as sp

from tools.symbolic.optype import BinaryOpType, BINARY_OPS
from discrete.fd.tools.stencil import AxStencil

from algebra.operator import Operator
from algebra.space import Space, ShapeTransform


class FDOperator(Operator):
    def __init__(
        self,
        space: Space,
        shape_transform: ShapeTransform,
        ax_stencils: tuple[AxStencil, ...],
    ):
        assert len(ax_stencils) == space.ndim
        super().__init__(space, shape_transform)
        self._ax_stencils = ax_stencils

    @property
    def stencils(self) -> tuple[AxStencil, ...]:
        return self._ax_stencils

    def copy(self) -> Self:
        stencils = tuple(stencil.copy() for stencil in self.stencils)
        return self.__class__(self.space, self.shape_transform, stencils)

    def modify(self, ax: int, new_stencil: AxStencil) -> Self:
        stencils = [stencil.copy() for stencil in self.stencils]
        stencils[ax] = new_stencil
        return self.__class__(self.space, self.shape_transform, tuple(stencils))

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        self.stencils[ax].eval_to(ax, inp, out)

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if not isinstance(other, type(self)):
            raise ValueError("FDOperator can obly be combined with other FDOperator")
        if other.shape_transform != self.shape_transform:
            raise ValueError("To combine operatros shape transforms have to match")

        stencils = []
        binary_op = BINARY_OPS[optype]
        for ax, stencil in enumerate(self.stencils):
            stencils.append(binary_op(stencil, other.stencils[ax]))
        return self.__class__(self.space, self.shape_transform, tuple(stencils))

    def _scale(self, other: float | int) -> Self:
        stencils = tuple(stencil * other for stencil in self.stencils)
        return self.__class__(self.space, self.shape_transform, stencils)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def as_array(self) -> sp.spmatrix:
        axis_1d = [
            self._axis_matrix(stencil, n)
            for stencil, n in zip(self.stencils, self.space.shape)
        ]
        per_axis_nd = self._extend_to_nd(axis_1d, self.space.shape)
        return self._combine_axes(per_axis_nd, self.shape_transform)

    def __neg__(self) -> Self:
        stencils = tuple(-stencil for stencil in self.stencils)
        return self.__class__(self.space, self.shape_transform, stencils)
