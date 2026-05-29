from typing import Self, Callable
import numpy as np
from algebra.core import Operator, FieldShape
from tools.stencil import AxStencil


class FDOperator(Operator):
    def __init__(
        self,
        input_shape: FieldShape,
        output_shape: FieldShape,
        ax_stencils: tuple[AxStencil, ...],
    ):
        assert len(ax_stencils) == input_shape.spacedim
        super().__init__(input_shape, output_shape)
        self._ax_stencils = ax_stencils

    @property
    def stencils(self) -> tuple[AxStencil, ...]:
        return self._ax_stencils

    def _new(self, stencils: tuple[AxStencil, ...]) -> Self:
        return self.__class__(self.input_shape, self.output_shape, tuple(stencils))

    def copy(self) -> Self:
        stencils = (stencil.copy() for stencil in self.stencils)
        return self._new(tuple(stencils))

    def modify(self, ax: int, new_stencil: AxStencil) -> Self:
        stencils = [stencil.copy() for stencil in self.stencils]
        stencils[ax] = new_stencil
        return self._new(tuple(stencils))

    def _combine(
        self,
        other: "FDOperator",
        binary_op: Callable[[AxStencil, AxStencil], AxStencil],
    ) -> Self:
        if not self.input_shape == other.input_shape:
            raise ValueError("Can only add StencilOperators of equal input shap")
        if not self.input_shape == other.input_shape:
            raise ValueError("Can only add StencilOperators of equal output shape")
        if not isinstance(other, type(self)):
            raise ValueError("Can only add StencilOperators of same type")
        stencils = []
        for ax, stencil in enumerate(self.stencils):
            stencils.append(binary_op(stencil, other.stencils[ax]))
        return self._new(tuple(stencils))

    def _scale(self, other: float | int) -> Self:
        stencils = tuple(stencil * other for stencil in self.stencils)
        return self._new(stencils)

    def __neg__(self) -> Self:
        return self._new(tuple(-s for s in self.stencils))

    def add(self, other: Self) -> Self:
        return self._combine(other, lambda a, b: a + b)

    def mul(self, other: Self) -> Self:
        return self._combine(other, lambda a, b: a * b)

    def div(self, other: Self) -> Self:
        return self._combine(other, lambda a, b: a / b)

    def scale(self, other: float) -> Self:
        return self._scale(other)

    def scale_arr(self, other: np.ndarray) -> Self:
        return NotImplemented
