from typing import Self

import numpy as np

from tools.symbolic.optype import BinaryOpType
from algebra.operator import Operator, ArrayOperator
from algebra.domain.bcs import BoundaryCondition
from algebra.domain.operator import DomainOperator

from discrete.fd.domain import FDDomain
from discrete.fd.stencil import StencilOperator
from .as_array import as_array


class FDDomainOperator(DomainOperator[FDDomain]):
    def __init__(self, domain: FDDomain, stencil: StencilOperator):
        super().__init__(domain, stencil.shape_transform)
        self._stencil = stencil

    def copy(self) -> Self:
        return self.__class__(self._domain, self._stencil.copy())

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        if isinstance(other, FDDomainOperator):
            return self.__class__(self._domain, self._stencil._combine(other._stencil, optype))
        return NotImplemented

    def _scale(self, other: float) -> Self:
        return self.__class__(self._domain, self._stencil._scale(other))

    def apply(self, inp: np.ndarray, out: np.ndarray):
        self._stencil.apply(inp, out)

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        pass

    def as_array(self) -> ArrayOperator:
        return as_array(self._stencil)

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        return self._domain.bc_tool.apply_bcs(bcs, self._stencil, rhs)
