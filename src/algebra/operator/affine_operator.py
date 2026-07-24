from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Any, Self, Generic
import numpy as np

from tools.symbolic import BinaryOpType, BINARY_OPS
from algebra.operator import Operator, TOperator
from algebra.expression import Expression
from algebra.expression.symbolic import SymbolicExpression


class AffineOperator(Operator, Generic[TOperator]):
    def __init__(
        self,
        operator: Operator,
        expression: Expression,
    ):
        self._operator = self._wrap_operator(operator)
        self._expression = self._wrap_expression(expression)
        super().__init__(operator.space, operator.shape_transform)

    @classmethod
    @abstractmethod
    def _wrap_operator(cls, operator: Operator) -> TOperator:
        pass

    @classmethod
    def _wrap_expression(cls, expression) -> SymbolicExpression:
        if not isinstance(expression, SymbolicExpression):
            return SymbolicExpression.wrap(expression)
        return expression

    @property
    def operator(self) -> TOperator:
        return self._operator

    @property
    def expression(self) -> SymbolicExpression:
        return self._expression

    def copy(self) -> Self:
        return self.__class__(self.operator, self.expression)

    def apply(self, inp: np.ndarray, out: np.ndarray):
        assert out.shape == self.expression.shape
        self.operator.apply(inp, out)
        out += self.expression.eval()

    def _scale(self, other: float) -> Self:
        return self.__class__(
            self.operator.copy() * other, self.expression.copy() * other
        )

    def _combine(self, other: Operator, optype: BinaryOpType) -> Self:
        binary_op = BINARY_OPS[optype]
        if isinstance(other, type(self)):
            return self._combine_affine(other, binary_op)
        return self._combine_operator(other, binary_op)

    def _combine_operator(
        self, other: Operator, binary_op: Callable[[Any, Any], Any]
    ) -> Self:
        return self.__class__(
            binary_op(self.operator.copy(), other.copy()), self.expression.copy()
        )

    def _combine_affine(
        self, other: "AffineOperator", binary_op: Callable[[Any, Any], Any]
    ) -> Self:
        return self.__class__(
            binary_op(self.operator.copy(), other.operator.copy()),
            binary_op(self.expression.copy(), other.expression.copy()),
        )

    def __neg__(self):
        return self.__class__(-self.operator.copy(), -self.expression.copy())
