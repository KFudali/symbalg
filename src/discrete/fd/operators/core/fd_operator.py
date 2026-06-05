from typing import Self

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
        return self.__class__(self.space, stencils)

    def modify(self, ax: int, new_stencil: AxStencil) -> Self:
        stencils = [stencil.copy() for stencil in self.stencils]
        stencils[ax] = new_stencil
        return self.__class__(self.space, tuple(stencils))

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if not isinstance(other, type(self)):
            raise ValueError("FDOperator can obly be combined with other FDOperator")
        stencils = []
        binary_op = BINARY_OPS[optype]
        for ax, stencil in enumerate(self.stencils):
            stencils.append(binary_op(stencil, other.stencils[ax]))
        return self.__class__(self.space, tuple(stencils))

    def _scale(self, other: float | int) -> Self:
        stencils = tuple(stencil * other for stencil in self.stencils)
        return self.__class__(self.space, stencils)

    def __neg__(self) -> Self:
        stencils = tuple(-stencil for stencil in self.stencils)
        return self.__class__(self.space, stencils)