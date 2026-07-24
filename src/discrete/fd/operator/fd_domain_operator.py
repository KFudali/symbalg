from typing import Self
import numpy as np


from tools.symbolic.optype import BinaryOpType
from algebra.operator import Operator
from algebra.domain import DomainOperator

from discrete.fd.domain import FDDomain
from discrete.fd.stencil import StencilOperator


class FDDomainOperator(DomainOperator[FDDomain]):
    def __init__(self, domain: FDDomain, stencil: StencilOperator):
        super().__init__(domain, stencil.shape_transform)
        self._stencil = stencil

    def copy(self) -> Self:
        return self.__class__(self._domain, self._stencil.copy())

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        return self.__class__(self._domain, self._stencil._combine(other, optype))

    def _scale(self, other: float) -> Self:
        return self.__class__(self._domain, self._stencil._scale(other))

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        self._stencil.apply(ax, inp, out)

    def as_array(self) -> ArrayOperator:
        pass

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        pass
