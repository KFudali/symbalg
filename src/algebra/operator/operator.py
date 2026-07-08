from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Self, TYPE_CHECKING
import numpy as np
from tools.symbolic import BinaryOpType
from algebra.space import Space, ShapeTransform, FieldShape
from .apply import APPLY

if TYPE_CHECKING:
    from algebra.field import Field
    from algebra.bcs import BoundaryCondition
    from algebra.expression import Expression
    from algebra.symbolic.array_operator import ArrayOperator


class Operator(ABC):
    def __init__(self, space: Space, shape_transform: ShapeTransform):
        self._space = space
        self._shape_transform = shape_transform
        self._apply_callable = APPLY[shape_transform]

    @property
    def shape_transform(self) -> ShapeTransform:
        return self._shape_transform

    @property
    def space(self) -> Space:
        return self._space

    def of(self, field: "Field") -> "Expression":
        from ..expression import CallableExpression

        def apply_to_field():
            return self.apply_to(field.value().eval())

        out_shape = self.shape_transform.transform(self.space, field.shape)
        return CallableExpression(
            FieldShape.from_shape(self.space, out_shape), apply_to_field
        )

    def apply(self, inp: np.ndarray, out: np.ndarray):
        self._apply_callable(self.space, self._apply, inp, out)

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
    def as_array(self) -> "ArrayOperator": ...

    @abstractmethod
    def apply_bcs(self, bcs: list[BoundaryCondition], rhs: np.ndarray) -> Self: ...

    @abstractmethod
    def copy(self) -> Self: ...

    @abstractmethod
    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray): ...

    @abstractmethod
    def _combine(self, other: Self, optype: BinaryOpType) -> Self: ...

    @abstractmethod
    def _scale(self, other: float) -> Self: ...

    @abstractmethod
    def __neg__(self) -> Self: ...

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

    def __radd__(self, other) -> Self:
        if isinstance(other, Operator):
            return self.combine(other, BinaryOpType.ADD)
        return NotImplemented

    def __rsub__(self, other) -> Self:
        if isinstance(other, Operator):
            return (self.__neg__()).combine(other, BinaryOpType.ADD)
        return NotImplemented


TOperator = TypeVar("TOperator", bound=Operator)
