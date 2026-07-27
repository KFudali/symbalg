from __future__ import annotations
from algebra.space import Space, SparseShape
from algebra.expression.symbolic.symbolic_expression import SymbolicExpression

from typing import Any, Self

import numpy as np
import scipy.sparse as sp

from algebra.expression import SparseExpression, ConstSparseExpression
from algebra.exceptions import ShapeMismatchError

from tools.symbolic import Symbolic, BinaryOpType, nodes
from .nodes import SparseExpressionNode


class SymbolicSparseExpression(Symbolic[SparseExpression], SparseExpression):
    def __init__(self, node: nodes.SymbolicNode[SparseExpression], shape: SparseShape):
        Symbolic.__init__(self, node)
        SparseExpression.__init__(self, shape)

    @classmethod
    def wrap(cls, value: SparseExpression) -> Self:
        node = cls._make_value(value)
        return cls(node, value.sparseshape)

    @classmethod
    def _make_value(cls, other: SparseExpression) -> SparseExpressionNode:
        return SparseExpressionNode(other)

    def _ensure_node(self, other: Any) -> SparseExpressionNode:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, nodes.SymbolicNode):
            return other
        if isinstance(other, sp.spmatrix):
            return SparseExpressionNode(ConstSparseExpression(self.space, other))
        return self._make_value(other)

    def eval(self) -> sp.spmatrix:
        return self.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node, self.sparseshape)

    def compatible(self, space: Space) -> bool:
        return self.resolve().compatible(space)

    def _new(self, node: nodes.SymbolicNode[SparseExpression]) -> Self:
        return self.__class__(node, self.sparseshape)

    def _combine_binary(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> Self:
        if not self._compatible(other, optype, reverse):
            return NotImplemented
        other_node = self._ensure_node(other)
        node = nodes.BinaryNode(optype, self.node, other_node)
        return self._new(node)

    def _compatible(
        self, other: Any, optype: BinaryOpType, reverse: bool = False
    ) -> bool:
        if isinstance(other, (int, float)):
            return True
        if isinstance(other, (sp.spmatrix, np.ndarray)):
            if other.shape in ((), self.shape):
                return True
            raise ShapeMismatchError(
                f"Incompatible shape: {other.shape} and {self.shape}"
            )
        if isinstance(other, SparseExpression):
            if other.shape == self.shape:
                return True
            raise ShapeMismatchError(
                f"Incompatible shape: {other.shape} and {self.shape}"
            )
        if isinstance(other, SymbolicSparseExpression):
            if other.shape == self.shape:
                return True
            raise ShapeMismatchError(
                f"Incompatible shape: {other.shape} and {self.shape}"
            )
        return True
