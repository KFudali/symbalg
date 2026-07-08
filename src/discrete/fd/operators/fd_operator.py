from typing import Self
import numpy as np

from tools.symbolic.optype import BinaryOpType, BINARY_OPS
from discrete.fd.tools.stencil import AxStencil
from discrete.fd.domain import FDDomain

from algebra.bcs import BoundaryCondition
from algebra.operator import Operator
from algebra.space import ShapeTransform
from algebra.symbolic import ArrayOperator
from algebra.systems import LinearSystem
from .utils import as_array


class FDOperator(Operator):
    def __init__(
        self,
        domain: FDDomain,
        shape_transform: ShapeTransform,
        ax_stencils: tuple[AxStencil, ...],
    ):
        assert len(ax_stencils) == domain.space.ndim
        super().__init__(domain.space, shape_transform)
        self._domain = domain
        self._ax_stencils = ax_stencils

    @property
    def stencils(self) -> tuple[AxStencil, ...]:
        return self._ax_stencils

    def copy(self) -> Self:
        stencils = tuple(stencil.copy() for stencil in self.stencils)
        return self.__class__(self._domain, self.shape_transform, stencils)

    def modify(self, ax: int, new_stencil: AxStencil) -> Self:
        stencils = [stencil.copy() for stencil in self.stencils]
        stencils[ax] = new_stencil
        return self.__class__(self._domain, self.shape_transform, tuple(stencils))

    def as_array(self) -> ArrayOperator:
        return ArrayOperator.from_array(
            self.space, self.shape_transform, as_array(self.space, self.stencils)
        )

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> LinearSystem:
        from .bcs import apply

        return apply(self._domain, bcs, LinearSystem(self, rhs))

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        self.stencils[ax].eval_to(ax, inp, out)

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if not isinstance(other, type(self)):
            raise ValueError(
                "FDOperator can obly be combined with other FDOperator or ArrayOperator"
            )
        if other.shape_transform != self.shape_transform:
            raise ValueError("To combine operatros shape transforms have to match")

        stencils = []
        binary_op = BINARY_OPS[optype]
        for ax, stencil in enumerate(self.stencils):
            stencils.append(binary_op(stencil, other.stencils[ax]))
        return self.__class__(self._domain, self.shape_transform, tuple(stencils))

    def _scale(self, other: float | int) -> Self:
        stencils = tuple(stencil * other for stencil in self.stencils)
        return self.__class__(self._domain, self.shape_transform, stencils)

    def __neg__(self) -> Self:
        stencils = tuple(-stencil for stencil in self.stencils)
        return self.__class__(self._space, self.shape_transform, stencils)
