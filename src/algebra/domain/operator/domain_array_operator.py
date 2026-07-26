from typing import Self, Generic

import numpy as np
from algebra.operator import ArrayOperator

from ..bcs import BoundaryCondition
from ..domain import TDomain
from .domain_operator import DomainOperator


class DomainArrayOperator(DomainOperator, Generic[TDomain]):
    def __init__(self, domain: TDomain, array_operator: ArrayOperator):
        super().__init__(domain.space, array_operator.shape_transform)
        self._array = array_operator

    @property
    def domain(self) -> TDomain:
        return self._domain

    def as_array(self) -> ArrayOperator:
        return self._array

    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self:
        return self._domain.bc_tool.apply_bcs_array(self._array, bcs, rhs)
