from typing import Self, Union
import numpy as np

from tools.symbolic.optype import BinaryOpType, BINARY_OPS
from discrete.fd.tools.stencil import AxStencil
from discrete.fd.domain import FDDomain

from algebra.bcs import BoundaryCondition
from algebra.operator import Operator
from algebra.space import Space, ShapeTransform
from algebra.symbolic import ArrayOperator
from algebra.systems import LinearSystem
from .utils import as_array

SpaceOrDomain = Union[Space, FDDomain]


class FDOperator(Operator):
    def __init__(
        self,
        space_or_domain: SpaceOrDomain,
        shape_transform: ShapeTransform,
        ax_stencils: tuple[AxStencil, ...],
    ):
        if isinstance(space_or_domain, FDDomain):
            domain = space_or_domain
            space = domain.space
        else:
            domain = None
            space = space_or_domain
        assert len(ax_stencils) == space.ndim
        super().__init__(space, shape_transform)
        self._domain = domain
        self._ax_stencils = ax_stencils

    def _domain_or_space(self) -> Space | FDDomain:
        return self._domain if self._domain is not None else self._space

    @property
    def stencils(self) -> tuple[AxStencil, ...]:
        return self._ax_stencils

    def copy(self) -> Self:
        stencils = tuple(stencil.copy() for stencil in self.stencils)
        return self.__class__(self._domain_or_space(), self.shape_transform, stencils)

    def modify(self, ax: int, new_stencil: AxStencil) -> Self:
        stencils = [stencil.copy() for stencil in self.stencils]
        stencils[ax] = new_stencil
        return self.__class__(
            self._domain_or_space(), self.shape_transform, tuple(stencils)
        )

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
        return self.__class__(
            self._domain_or_space(), self.shape_transform, tuple(stencils)
        )

    def _scale(self, other: float | int) -> Self:
        stencils = tuple(stencil * other for stencil in self.stencils)
        return self.__class__(self._domain_or_space(), self.shape_transform, stencils)

    def __neg__(self) -> Self:
        stencils = tuple(-stencil for stencil in self.stencils)
        return self.__class__(self._domain_or_space(), self.shape_transform, stencils)
