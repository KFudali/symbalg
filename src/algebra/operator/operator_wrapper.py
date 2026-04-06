from __future__ import annotations
from typing import Self

import numpy as np
from .operator import Operator


class OperatorWrapper(Operator):
    def __init__(self, operator: Operator):
        super().__init__(operator.input_shape, operator.output_shape)
        self._operator = operator

    @property
    def operator(self) -> Operator:
        return self._operator

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        self._operator.apply(input_field, output_field)

    def _new(self, operator: Operator) -> Self:
        return OperatorWrapper(operator)

    def __neg__(self) -> Self:
        return self._new(-self.operator)

    def add(self, other: Operator):
        return self._new(self.operator.add(other))

    def mul(self, other: Operator):
        return self._new(self.operator.mul(other))

    def scale(self, other: float):
        return self._new(self.operator.scale(other))

    def scale_arr(self, other: float):
        return self._new(self.operator.scale_arr(other))
