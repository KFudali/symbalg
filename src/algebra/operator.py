from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar
from .expression import Expression
from .space import Space, ShapeTransform


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

    @abstractmethod
    def apply(self, expression: Expression) -> Expression:
        pass

    @abstractmethod
    def _combine(self, other: "Operator", binary_op):
        pass

    @abstractmethod
    def _scale(self, other: float) -> "Operator":
        pass

    def combine(self, other: "Operator", binary_op) -> "Operator":
        assert (
            other.space == self.space
        ), "Cannot combine operators from different spaces"
        assert (
            other.shape_transform == self.shape_transform
        ), "Cannot combine operators with different shape transformations"
        return self._combine(other, binary_op)

    def __neg__(self) -> Operator:
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Operator):
            return self.combine(other, lambda a, b: a + b)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Operator):
            return self.combine(other, lambda a, b: a - b)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Operator):
            return self.combine(other, lambda a, b: a * b)
        if isinstance(other, float):
            return self._scale(other)
        return NotImplemented

    def __div__(self, other):
        if isinstance(other, Operator):
            return self.combine(other, lambda a, b: a / b)
        if isinstance(other, float):
            return self._scale(1.0 / other)
        return NotImplemented

    def __rmul__(self, other: float):
        return self.__mul__(other)


TOperator = TypeVar("TOperator", bound=Operator)
