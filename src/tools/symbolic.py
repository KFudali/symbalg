from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union
from dataclasses import dataclass
from enum import Enum, auto

TOperand = TypeVar("TOperand")

class Op(ABC, Generic[TOperand]):
    @abstractmethod
    def fold(self) -> TOperand:
        pass

@dataclass
class Value(Op[TOperand]):
    value: TOperand

    def fold(self) -> TOperand:
        return self.value

@dataclass
class UnaryOp(Op[TOperand]):
    class OpType(Enum):
        NEG = auto()

    optype: OpType
    operand: Op[TOperand]

    def fold(self) -> TOperand:
        val = self.operand.fold()

        if self.optype == UnaryOp.OpType.NEG:
            return -val  # type: ignore

        raise ValueError(f"Unknown OpType: {self.optype}")

@dataclass
class BinaryOp(Op[TOperand]):
    class OpType(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()

    optype: OpType
    left: Op[TOperand]
    right: Op[TOperand]

    def fold(self) -> TOperand:
        l = self.left.fold()
        r = self.right.fold()

        match self.optype:
            case BinaryOp.OpType.ADD:
                return l + r
            case BinaryOp.OpType.SUB:
                return l - r
            case BinaryOp.OpType.MUL:
                return l * r 
            case BinaryOp.OpType.DIV:
                return l / r

        raise ValueError(f"Unknown OpType: {self.optype}")

SymbolicNode = Union[
    TOperand,
    Op[TOperand],
    "Symbolic[TOperand]",
]

class Symbolic(Op[TOperand], Generic[TOperand]):
    def __init__(self, init_node: SymbolicNode[TOperand]):
        if isinstance(init_node, Symbolic):
            self._node: Op[TOperand] = init_node.node
        elif isinstance(init_node, Op):
            self._node = init_node
        else:
            self._node = Value(init_node)

    @property
    def node(self) -> SymbolicNode:
        return self._node

    def _wrap(self, other: SymbolicNode[TOperand]) -> Op[TOperand]:
        if isinstance(other, Symbolic):
            return other.node
        if isinstance(other, Op):
            return other
        return Value(other)

    def _new(self, expr: Op[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(expr)

    def _make_binary(self, other: SymbolicNode, optype: BinaryOp.OpType) -> BinaryOp:
        return BinaryOp(optype, self._node, self._wrap(other))

    def _make_unary(self, optype: UnaryOp.OpType) -> BinaryOp:
        return UnaryOp(optype, self._node)

    # ---- operations ----

    def neg(self) -> Symbolic[TOperand]:
        return self._new(self._make_unary(UnaryOp.OpType.NEG))

    def add(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(self._make_binary(other, BinaryOp.OpType.ADD))

    def sub(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(self._make_binary(other, BinaryOp.OpType.SUB))

    def mul(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(self._make_binary(other, BinaryOp.OpType.MUL))

    def div(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(self._make_binary(other, BinaryOp.OpType.DIV))

    # ---- operator overloads ----
    def __neg__(self) -> Symbolic[TOperand]:
        return self.neg()

    def __add__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self.add(other)

    def __sub__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self.sub(other)

    def __mul__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self.mul(other)

    def __truediv__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self.div(other)

    # ---- reverse operators ----

    def __radd__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(other).add(self)

    def __rsub__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(other).sub(self)

    def __rmul__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(other).mul(self)

    def __rtruediv__(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(other).div(self)

    # ---- evaluation ----

    def fold(self) -> TOperand:
        return self._node.fold()

    def __repr__(self) -> str:
        return f"Symbolic({self._node})"

