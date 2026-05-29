from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Self, Union
from .nodes import SymbolicNode, ValueNode, UnaryNode, BinaryNode, TSymbolic
from .optype import UnaryOpType, BinaryOpType


@dataclass(frozen=True)
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

    def _new(self, expr: SymbolicNode[TSymbolic]) -> Self:
        return self.__class__(expr)

    def _make_binary(
        self, other: SymbolicNode, optype: BinaryOpType
    ) -> BinaryNode[TSymbolic]:
        self._assert_compatible(other, optype)
        return BinaryNode(optype, self._node, self._ensure_node(other))

    def _make_unary(self, optype: UnaryOpType) -> UnaryNode[TSymbolic]:
        return UnaryNode(optype, self._node)

    def _make_value(self, operand: TSymbolic) -> ValueNode[TSymbolic]:
        return ValueNode(operand)

    # ---- operator overloads ----
    def __neg__(self) -> Self:
        return self._new(self._make_unary(UnaryOpType.NEG))

    def __add__(self, other: Types) -> Self:
        return self._new(self._make_binary(other, BinaryOpType.ADD))

    def __sub__(self, other: Types) -> Self:
        return self._new(self._make_binary(other, BinaryOpType.ADD))

    def __mul__(self, other: Types) -> Self:
        return self._new(self._make_binary(other, BinaryOpType.ADD))

    def __truediv__(self, other: Types) -> Self:
        return self._new(self._make_binary(other, BinaryOpType.ADD))

    # ---- reverse operators ----

    def __radd__(self, other: Types) -> Self:
        self._assert_compatible(other, BinaryOpType.ADD)
        return self.__add__(other)

    def __rsub__(self, other: Types) -> Self:
        return self.__neg__().__add__(other)

    def __rmul__(self, other: Types) -> Self:
        return self.__mul__(other)

    def __rtruediv__(self, other: Types) -> Self:
        return self._new(self._ensure_node(other)).__truediv__(self._node)

    def __repr__(self) -> str:
        return f"Symbolic({self.node})"
