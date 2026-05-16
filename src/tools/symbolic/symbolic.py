from __future__ import annotations
from typing import Any, Union
from .symbolic_node import SymbolicNode, TSymbolic, ValueNode, UnaryNode, BinaryNode
from .optype import UnaryOpType, BinaryOpType


class Symbolic(SymbolicNode[TSymbolic]):
    Types = Union["Symbolic", SymbolicNode[TSymbolic], TSymbolic]

    def __init__(self, node: Types):
        self._node = self._ensure_node(node)

    @property
    def node(self) -> SymbolicNode[TSymbolic]:
        return self._node

    def resolve(self) -> TSymbolic:
        return self._node.resolve()

    def _ensure_node(self, other) -> SymbolicNode[TSymbolic]:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, SymbolicNode):
            return other
        return self._make_value(other)

    def _assert_compatible(self, other: Any, optype: BinaryOpType):
        pass

    def _new(self, expr: SymbolicNode[TSymbolic]) -> "Symbolic[TSymbolic]":
        return Symbolic(expr)

    def _make_binary(
        self, other: SymbolicNode, optype: BinaryOpType
    ) -> BinaryNode[TSymbolic]:
        return BinaryNode(optype, self._node, self._ensure_node(other))

    def _make_unary(self, optype: UnaryOpType) -> UnaryNode[TSymbolic]:
        return UnaryNode(optype, self._node)

    def _make_value(self, operand: TSymbolic) -> ValueNode[TSymbolic]:
        return ValueNode(operand)

    # ---- operations ----

    def neg(self) -> "SymbolicNode[TSymbolic]":
        return self._new(self._make_unary(UnaryOpType.NEG))

    def add(self, other: SymbolicNode[TSymbolic]) -> "Symbolic[TSymbolic]":
        return self._new(self._make_binary(other, BinaryOpType.ADD))

    def sub(self, other: SymbolicNode[TSymbolic]) -> "Symbolic[TSymbolic]":
        return self._new(self._make_binary(other, BinaryOpType.SUB))

    def mul(self, other: SymbolicNode[TSymbolic]) -> "Symbolic[TSymbolic]":
        return self._new(self._make_binary(other, BinaryOpType.MUL))

    def div(self, other: SymbolicNode[TSymbolic]) -> "Symbolic[TSymbolic]":
        return self._new(self._make_binary(other, BinaryOpType.DIV))

    # ---- operator overloads ----
    def __neg__(self) -> "SymbolicNode[TSymbolic]":
        return self.neg()

    def __add__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.ADD)
        return self.add(other)

    def __sub__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.SUB)
        return self.sub(other)

    def __mul__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.MUL)
        return self.mul(other)

    def __truediv__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.DIV)
        return self.div(other)

    # ---- reverse operators ----

    def __radd__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.DIV)
        return self._new(self.add(other))

    def __rsub__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.SUB)
        return self._new(other) + (-self)

    def __rmul__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.MUL)
        return self._new(self.mul(other))

    def __rtruediv__(self, other: Types) -> "Symbolic[TSymbolic]":
        self._assert_compatible(other, BinaryOpType.DIV)
        return self._new(other).div(self)

    def __repr__(self) -> str:
        return f"Symbolic({self.node})"
