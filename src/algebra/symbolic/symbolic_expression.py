from __future__ import annotations
from algebra.expression import Expression
from tools import Symbolic


class SymbolicExpression(Expression, Symbolic[Expression]):
    def __init__(self, expression: Expression | "SymbolicExpressoin"):
        Expression.__init__(self, expression.output_shape)
        Symbolic[Expression].__init__(self, Expression)
