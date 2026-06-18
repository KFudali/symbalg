from __future__ import annotations
from typing import Any, Self
import numpy as np

from algebra.expression import Expression, ScalarExpression, CallableExpression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
from .nodes import ExpressionNode


class SymbolicExpression(Symbolic[Expression], Expression):
    def __init__(self, node: nodes.SymbolicNode[Expression], shape: tuple[int, ...]):
        Symbolic.__init__(self, node)
        Expression.__init__(self, shape)

    @classmethod
    def wrap(cls, value: Expression) -> Self:
        node = cls._make_value(value)
        return cls(node, value.shape)

    @classmethod
    def _make_value(cls, other: Expression) -> nodes.ValueNode[Expression]:
        expr = other
        if isinstance(other, float):
            expr = ScalarExpression(other)
        if isinstance(other, np.ndarray):

            def get_other() -> np.ndarray:
                return other

            expr = CallableExpression(other.shape, get_other)
        return ExpressionNode(expr)

    def eval(self) -> np.ndarray:
        return self.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node, self.shape)

    def _new(self, node: nodes.SymbolicNode[Expression]) -> Self:
        return self.__class__(node, self.shape)

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        if isinstance(other, float):
            return True
        if isinstance(other, (Expression, np.ndarray)):
            if self.shape in ((), other.shape) or other.shape == ():
                return True
            raise ShapeMismatchError(
                f"Incompatible shapes: {self.shape} and {other.shape}"
            )
        return False
