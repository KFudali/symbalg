from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Self
from .nodes import SymbolicNode, ValueNode, UnaryNode, BinaryNode, TSymbolic
from .optype import UnaryOpType, BinaryOpType


@dataclass(frozen=True)
class Symbolic(SymbolicNode[TSymbolic]):
    node: SymbolicNode[TSymbolic]

    def resolve(self) -> TSymbolic:
        return self.node.resolve()

    def copy(self) -> Self:
        return self.__class__(self.node)

    @classmethod
    def wrap(cls, value: TSymbolic) -> Self:
        return cls(cls._make_value((value)))

    @classmethod
    def _ensure_node(
        cls, other: Symbolic | SymbolicNode | TSymbolic
    ) -> SymbolicNode[TSymbolic]:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, SymbolicNode):
            return other
        return cls._make_value(other)

    @classmethod
    def _make_value(cls, other: TSymbolic) -> ValueNode[TSymbolic]:
        return ValueNode(other)

    def _new(self, node: SymbolicNode[TSymbolic]) -> Self:
        return self.__class__(node)

    def _compatible(self, other: Any, optype: BinaryOpType) -> bool:
        return True

    def _combine_binary(self, other: Any, optype: BinaryOpType) -> Self:
        if self._compatible(other, optype):
            other_node = self._ensure_node(other)
            return self._new(BinaryNode(optype, self.node, other_node))
        return NotImplemented

    # ---- operator overloads ----
    def __neg__(self) -> Self:
        return self._new(UnaryNode(UnaryOpType.NEG, self))

    def __add__(self, other: Any) -> Self:
        return self._combine_binary(other, BinaryOpType.ADD)

    def __sub__(self, other: Any) -> Self:
        return self._combine_binary(other, BinaryOpType.SUB)

    def __mul__(self, other: Any) -> Self:
        return self._combine_binary(other, BinaryOpType.MUL)

    def __truediv__(self, other: Any) -> Self:
        return self._combine_binary(other, BinaryOpType.DIV)

    # ---- reverse operators ----
    def __radd__(self, other: Any) -> Self:
        return self.__add__(other)

    def __rsub__(self, other: Any) -> Self:
        return self.__neg__().__add__(other)

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    def __rtruediv__(self, other: Any) -> Self:
        return self._new(self._ensure_node(other)).__truediv__(self.node)

    def __repr__(self) -> str:
        return f"Symbolic({self.node})"
