from abc import ABC, abstractmethod
from typing import Self, Generic

import numpy as np
from algebra.expression import CallableExpression, symbolic
from algebra.operator import Operator, ArrayOperator
from algebra.space import ShapeTransform, FieldShape
from algebra.field import Field

from ..bcs import BoundaryCondition
from ..domain import TDomain


class DomainOperator(Operator, ABC, Generic[TDomain]):
    def __init__(self, domain: TDomain, shape_transform: ShapeTransform):
        super().__init__(domain.space, shape_transform)
        self._domain = domain

    @property
    def domain(self) -> TDomain:
        return self._domain

    @abstractmethod
    def as_array(self) -> ArrayOperator: ...

    @abstractmethod
    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self: ...

    def of(self, field: Field) -> symbolic.SymbolicExpression:
        def apply_to_field():
            return self.apply_to(field.value().eval())

        out_shape = self.shape_transform.transform(self.space, field.shape)
        result = CallableExpression(
            FieldShape.from_shape(self.space, out_shape), apply_to_field
        )
        return symbolic.SymbolicExpression.wrap(result)
