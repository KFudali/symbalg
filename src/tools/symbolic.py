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

class Symbolic(Generic[TOperand]):
    def __init__(self, init_operand: SymbolicNode[TOperand]):
        if isinstance(init_operand, Symbolic):
            self._expr: Op[TOperand] = init_operand._expr
        elif isinstance(init_operand, Op):
            self._expr = init_operand
        else:
            self._expr = Value(init_operand)

    def _wrap(self, other: SymbolicNode[TOperand]) -> Op[TOperand]:
        if isinstance(other, Symbolic):
            return other._expr
        if isinstance(other, Op):
            return other
        return Value(other)

    def _new(self, expr: Op[TOperand]) -> Symbolic[TOperand]:
        return Symbolic(expr)

    # ---- operations ----

    def neg(self) -> Symbolic[TOperand]:
        return self._new(UnaryOp(UnaryOp.OpType.NEG, self._expr))

    def add(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(BinaryOp(BinaryOp.OpType.ADD, self._expr, self._wrap(other)))

    def sub(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(BinaryOp(BinaryOp.OpType.SUB, self._expr, self._wrap(other)))

    def mul(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(BinaryOp(BinaryOp.OpType.MUL, self._expr, self._wrap(other)))

    def div(self, other: SymbolicNode[TOperand]) -> Symbolic[TOperand]:
        return self._new(BinaryOp(BinaryOp.OpType.DIV, self._expr, self._wrap(other)))

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
        return self._expr.fold()

    def __repr__(self) -> str:
        return f"Symbolic({self._expr})"