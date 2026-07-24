from __future__ import annotations

from typing import Self

import numpy as np

from tools.symbolic.optype import BinaryOpType, BINARY_OPS
from algebra.operator import Operator
from algebra.space import Space, ShapeTransform
from .ax_stencil import AxStencil


class StencilOperator(Operator):
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
