from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Self, TYPE_CHECKING
import numpy as np
from tools.symbolic.optype import BinaryOpType
from .space import Space, ShapeTransform

if TYPE_CHECKING:
    from .field import Field
    from .expression import Expression


class Operator(ABC):
    def __init__(self, space: Space, shape_transform: ShapeTransform):
        self._space = space
        self._shape_transform = shape_transform

    @property
    def shape_transform(self) -> ShapeTransform:
        return self._shape_transform

    @property
    def space(self) -> Space:
        return self._space

    def of(self, field: "Field") -> "Expression":
        from .expression import CallableExpression

        def apply_to_field():
            return self.apply_to(field.value().eval())

        out_shape = self.shape_transform.transform(self.space, field.shape)
        return CallableExpression(out_shape, apply_to_field)

    @abstractmethod
    def copy(self) -> Self:
        pass

    @abstractmethod
    def apply(self, inp: np.ndarray, out: np.ndarray):
        pass

    def apply_to(self, inp: np.ndarray) -> np.ndarray:
        out_shape = self._shape_transform.transform(self._space, inp.shape)
        out = np.zeros(out_shape, dtype=inp.dtype)
        self.apply(inp, out)
        return out

    def combine(self, other: Self, optype: BinaryOpType) -> Self:
        assert (
            other.space == self.space
        ), "Cannot combine operators from different spaces"
        assert (
            other.shape_transform == self.shape_transform
        ), "Cannot combine operators with different shape transformations"
        return self._combine(other, optype)

    @abstractmethod
    def _combine(self, other: Self, optype: BinaryOpType) -> Self:
        pass

    @abstractmethod
    def _scale(self, other: float) -> Self:
        pass

    @abstractmethod
    def __neg__(self) -> Self:
        return NotImplemented

    def __add__(self, other) -> Self:
        if isinstance(other, Operator):
            return self.combine(other, BinaryOpType.ADD)
        return NotImplemented

    def __sub__(self, other) -> Self:
        if isinstance(other, Operator):
            return self.combine(other, BinaryOpType.SUB)
        return NotImplemented

    def __mul__(self, other) -> Self:
        if isinstance(other, Operator):
            return self.combine(other, BinaryOpType.MUL)
        if isinstance(other, float):
            return self._scale(other)
        return NotImplemented

    def __truediv__(self, other) -> Self:
        if isinstance(other, Operator):
            return self.combine(other, BinaryOpType.DIV)
        if isinstance(other, float):
            return self._scale(1.0 / other)
        return NotImplemented

    def __rmul__(self, other: float) -> Self:
        return self.__mul__(other)


TOperator = TypeVar("TOperator", bound=Operator)
