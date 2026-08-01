from __future__ import annotations
from abc import ABC, abstractmethod
import scipy.sparse as sp
from sparse import SparseArray, COO
from tools.buffer import (
    ValueBuffer,
    ShiftProxyValueBuffer,
    ComponentProxyValueBuffer,
    StackedProxyValueBuffer,
)
from tools.action import LazyAction

from .operator import ArrayOperator
from .expression import (
    Expression,
    FieldExpression,
    CallableExpression,
)
from .expression.symbolic import SymbolicExpression
from .space import SpaceShaped, Shape, ShapeTransform
from .space import shape_utils as utils


class AbstractField(SpaceShaped, ABC):
    @abstractmethod
    def past(self, step: int) -> "AbstractField":
        pass

    @abstractmethod
    def value(self) -> FieldExpression:
        pass

    @abstractmethod
    def set_value(self, value: Expression) -> LazyAction:
        pass


class Field(AbstractField):
    def __init__(self, shape: Shape, value_buffer: ValueBuffer):
        assert shape.fieldshape() == value_buffer.shape
        super().__init__(shape)
        self._value_buffer = value_buffer

    @property
    def buffer(self) -> ValueBuffer:
        return self._value_buffer

    def past(self, step: int) -> "Field":
        return Field(self.shape, ShiftProxyValueBuffer(self._value_buffer, step))

    def value(self) -> FieldExpression:
        return SymbolicExpression.wrap(
            CallableExpression(self.shape, self._value_buffer.get)
        )

    def set_value(self, value: Expression) -> LazyAction:
        assert value.shape == self.shape

        def _set_value() -> None:
            self._value_buffer.set(value.eval())

        return LazyAction(_set_value)

    def component(self, comp_query: int | tuple[slice | int, ...]) -> "Field":
        query = utils.pick_component(self.shape, comp_query)
        buffer = ComponentProxyValueBuffer(self._value_buffer, query)
        result_shape = Shape(self.space, buffer.shape[: -self.space.ndim])
        return Field(result_shape, buffer)


def stack(fields: tuple[Field, ...], ax: int = 0) -> Field:
    try:
        buffer = StackedProxyValueBuffer(tuple(field.buffer for field in fields))
    except Exception as e:
        raise ValueError(f"Cannot stack fields over ax: {ax}") from e
    space = fields[0].space
    shape = Shape(space, buffer.shape[: -space.ndim])
    return Field(shape, buffer)


def to_operator(
    field: Field, shape_transform: ShapeTransform = ShapeTransform.NONE
) -> ArrayOperator:
    def _diag() -> SparseArray:
        return COO(sp.diags(field.value().eval().ravel(), 0))

    mat = CallableExpression(Shape(field.space, ()), _diag)
    return ArrayOperator(field.space, shape_transform, mat)
