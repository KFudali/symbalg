from __future__ import annotations
from typing import Any, Self, TypeVar
import numpy as np

from algebra.space import Space, ShapeTransform
from algebra.exceptions import ShapeMismatchError

from algebra.expression import Expression
from algebra.expression.symbolic.nodes import ExpressionNode

from tools.symbolic import Symbolic, BinaryOpType, nodes

from .core import Operator, TOperator


class SymbolicOperator(Symbolic[TOperator], Operator):
    def __init__(
        self,
        node: nodes.SymbolicNode[Operator],
        space: Space,
        shape_transform: ShapeTransform,
    ):
        Symbolic.__init__(self, node)
        Operator.__init__(self, space, shape_transform)

    def apply(self, inp: np.ndarray, out: np.ndarray):
        self.resolve().apply(inp, out)

    def _apply(self, ax: int, inp: np.ndarray, out: np.ndarray):
        pass

    def _scale(self, other: float) -> Self:
        raise ValueError("SymbolicOperator should not use _scale method")

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        raise ValueError("SymbolicOperator should not use _combine method")

    @classmethod
    def wrap(cls, value: TOperator) -> Self:
        node = cls._make_value(value)
        return cls(node, value.space, value.shape_transform)

    def copy(self) -> Self:
        return self.__class__(self.node, self.space, self.shape_transform)

    @classmethod
    def _make_value(cls, other: Any) -> nodes.ValueNode:
        if isinstance(other, Expression):
            if other.shape.is_scalar():
                return ExpressionNode(other)
        return super()._make_value(other)

    def _new(self, node: nodes.SymbolicNode[TOperator]) -> Self:
        return self.__class__(node, self.space, self.shape_transform)

    def _compatible(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> bool:
        if isinstance(other, Operator):
            if self.space != other.space:
                raise ShapeMismatchError(
                    f"Incompatible space: {self.space} and {other.space}"
                )
            if self.shape_transform != other.shape_transform:
                raise ShapeMismatchError(
                    f"Incompatible shape_transform: {self.shape_transform} "
                    f"and {other.shape_transform}"
                )
            return True
        if reverse and optype == BinaryOpType.DIV:
            # cannot divide by operator
            return False
        is_scale = optype in (BinaryOpType.DIV, BinaryOpType.MUL)
        if isinstance(other, Expression):
            if other.shape.is_scalar() and is_scale:
                return True
            raise ShapeMismatchError(f"Incompatible expression shape: {other.shape}")
        if isinstance(other, float):
            if is_scale:
                return True
        return False


TSymbolicOperator = TypeVar("TSymbolicOperator", bound=SymbolicOperator)
