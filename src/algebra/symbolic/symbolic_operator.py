from __future__ import annotations
from typing import Any, Self, TYPE_CHECKING
import numpy as np
import scipy.sparse as sp

from algebra.space import Space, ShapeTransform
from algebra.operator import Operator, TOperator
from algebra.expression import Expression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
from .nodes import ExpressionNode

if TYPE_CHECKING:
    from algebra.field import Field
    from .symbolic_expression import SymbolicExpression


class SymbolicOperator(Symbolic[TOperator], Operator):
    def __init__(
        self,
        node: nodes.SymbolicNode[Operator],
        space: Space,
        shape_transform: ShapeTransform,
    ):
        Symbolic.__init__(self, node)
        Operator.__init__(self, space, shape_transform)

    def of(self, field: "Field") -> "SymbolicExpression":
        from .symbolic_expression import SymbolicExpression

        return SymbolicExpression.wrap(Operator.of(self, field))

    def apply(self, inp: np.ndarray, out: np.ndarray):
        self.resolve().apply(inp, out)

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
            if other.fieldshape.is_scalar():
                return ExpressionNode(other)
        return super()._make_value(other)

    def _new(self, node: nodes.SymbolicNode[TOperator]) -> Self:
        return self.__class__(node, self.space, self.shape_transform)

    def as_array(self) -> sp.spmatrix:
        return self.resolve().as_array()

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
            if other.shape == () and is_scale:
                return True
            raise ShapeMismatchError(f"Incompatible expression shape: {other.shape}")
        if isinstance(other, float):
            if is_scale:
                return True
        return False
