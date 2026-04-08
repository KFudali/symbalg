from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union, Self, Any
from dataclasses import dataclass
from enum import Enum, auto

TOperand = TypeVar("TOperand")

class Op(ABC, Generic[TOperand]):
    @abstractmethod
    def fold(self) -> TOperand:
        pass

@dataclass(frozen=True)
class Value(Op[TOperand]):
    value: TOperand

    def fold(self) -> TOperand:
        return self.value

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

class Symbolic(Op[TOperand]):
    def __init__(self, init_node: SymbolicNode[TOperand]):
        self._base_op = self._wrap(init_node)
    
    @property
    def base_op(self) -> SymbolicNode:
        return self._base_op

    def _wrap(self, other: SymbolicNode[TOperand]) -> Op[TOperand]:
        if isinstance(other, Symbolic):
            return other.base_op
        if isinstance(other, Op):
            return other
        return self._make_value(other)

    def _assert_compatible(self, other: Any):
        pass

    def _new(self, expr: Op[TOperand]) -> Self:
        return Symbolic(expr)

    def _make_binary(
        self,
        other: SymbolicNode,
        optype: BinaryOp.OpType
    ) -> BinaryOp[TOperand]:
        return BinaryOp(optype, self._base_op, self._wrap(other))

    def _make_unary(self, optype: UnaryOp.OpType) -> UnaryOp[TOperand]:
        return UnaryOp(optype, self._base_op)

    def _make_value(self, operand: TOperand) -> Value[TOperand]:
        return Value(operand)

    # ---- operations ----

    def neg(self) -> Self:
        return self._new(self._make_unary(UnaryOp.OpType.NEG))

    def add(self, other: SymbolicNode[TOperand]) -> Self:
        return self._new(self._make_binary(other, BinaryOp.OpType.ADD))

    def sub(self, other: SymbolicNode[TOperand]) -> Self:
        return self._new(self._make_binary(other, BinaryOp.OpType.SUB))

    def mul(self, other: SymbolicNode[TOperand]) -> Self:
        return self._new(self._make_binary(other, BinaryOp.OpType.MUL))

    def div(self, other: SymbolicNode[TOperand]) -> Self:
        return self._new(self._make_binary(other, BinaryOp.OpType.DIV))

    # ---- operator overloads ----
    def __neg__(self) -> Self:
        return self.neg()

    def __add__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return self.add(other)

    def __sub__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return self.sub(other)

    def __mul__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return self.mul(other)

    def __truediv__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return self.div(other)

    # ---- reverse operators ----

    def __radd__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return Symbolic(other).add(self)
    
    def __rsub__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return Symbolic(other).sub(self)

    def __rmul__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return Symbolic(other).mul(self)
    
    def __rtruediv__(self, other: SymbolicNode[TOperand]) -> Self:
        self._assert_compatible(other)
        return Symbolic(other).div(self)

    def fold(self) -> TOperand:
        return self._base_op.fold()

    def __repr__(self) -> str:
        return f"Symbolic({self._base_op})"
