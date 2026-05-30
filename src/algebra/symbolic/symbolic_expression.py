from __future__ import annotations
from typing import Any, Self

from algebra.expression import Expression

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
        return ExpressionNode(other)

    def copy(self) -> Self:
        return self.__class__(self.node, self.shape)

    def _new(self, node: nodes.SymbolicNode[Expression]) -> Self:
        return self.__class__(node, self.shape)

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        is_scale = optype in (BinaryOpType.DIV, BinaryOpType.MUL)
        if isinstance(other, Expression):
            if self.shape == other.shape:
                return True
        if isinstance(other, float):
            if is_scale:
                return True
        return False
