from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, TypeAlias
from dataclasses import dataclass

from enum import Enum, auto

TOperand = TypeVar("TOperand")

class Op(ABC, Generic[TOperand]):
    @abstractmethod
    def fold(self) -> TOperand:
        pass

@dataclass
class UnaryOp(Op[TOperand], Generic[TOperand]):
    class OpType(Enum):
        NEG = auto()
    NEG = OpType.NEG

    optype: UnaryOp.OpType
    operand: TOperand

    def fold(self) -> TOperand:
        if self.optype == UnaryOp.NEG:
            return -self.operand
        raise ValueError(f"Unknown OpType: {self.optype}")

@dataclass
class BinaryOp(Op[TOperand],Generic[TOperand]):
    class OpType(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
    ADD = OpType.ADD
    SUB = OpType.SUB
    MUL = OpType.MUL
    DIV = OpType.DIV

    optype: BinaryOp.OpType
    left: TOperand
    right: TOperand

    def fold(self) -> TOperand:
        match self.optype:
            case BinaryOp.ADD:
                return self.left + self.right
            case BinaryOp.SUB:
                return self.left - self.right
            case BinaryOp.MUL:
                return self.left * self.right
            case BinaryOp.DIV:
                return self.left / self.right
        raise ValueError(f"Unknown OpType: {self.optype}")


class Symbolic(Generic[TOperand]):
    def __init__(
        self,
        init_operand: TOperand | Op,
        make_unary: Callable[[UnaryOp.OpType, TOperand], UnaryOp] = UnaryOp,
        make_binary: Callable[[BinaryOp.OpType, TOperand, TOperand], BinaryOp] = BinaryOp
    ):
        self._base = init_operand
        self._make_unary = make_unary
        self._make_binary = make_binary

    def _new(self, init_operand: TOperand | Op) -> Symbolic[TOperand]:
        return Symbolic(init_operand, self._make_binary, self._make_binary)

    def neg(self) -> Symbolic[TOperand]:
        return self._new(self._make_unary(UnaryOp.NEG, self._base))

    def add(self, other: TOperand) -> Symbolic[TOperand]:
        return self._new(self._make_binary(BinaryOp.ADD, self._base, other))

    def sub(self, other: TOperand) -> Symbolic[TOperand]:
        return self._new(self._make_binary(BinaryOp.SUB, self._base, other))

    def mul(self, other: TOperand) -> Symbolic[TOperand]:
        return self._new(self._make_binary(BinaryOp.MUL, self._base, other))

    def div(self, other: TOperand) -> Symbolic[TOperand]:
        return self._new(self._make_binary(BinaryOp.DIV, self._base, other))

    def __neg__(self) -> Symbolic[TOperand]:
        return self.neg()

    def __add__(self, other: TOperand) -> Symbolic[TOperand]:
        return self.add(other)

    def __sub__(self, other: TOperand) -> Symbolic[TOperand]:
        return self.sub(other)

    def __mul__(self, other: TOperand) -> Symbolic[TOperand]:
        return self.mul(other)

    def __truediv__(self, other: TOperand) -> Symbolic[TOperand]:
        return self.div(other)

    def fold(self) -> TOperand:
        if isinstance(self._base, Op):
            return self._base.fold()
        return self._base

    def __repr__(self) -> str:
        return f"Symbolic({self._base!r})"
