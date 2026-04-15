from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
from .optype import UnaryOpType, BinaryOpType

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
    optype: UnaryOpType
    operand: Op[TOperand]

    def fold(self) -> TOperand:
        val = self.operand.fold()

        if self.optype == UnaryOpType.NEG:
            return -val

        raise ValueError(f"Unknown OpType: {self.optype}")

@dataclass(frozen=True)
class BinaryOp(Op[TOperand]):
    optype: BinaryOpType
    left: Op[TOperand]
    right: Op[TOperand]

    def fold(self) -> TOperand:
        l = self.left.fold()
        r = self.right.fold()

        match self.optype:
            case BinaryOpType.ADD:
                return l + r
            case BinaryOpType.SUB:
                return l - r
            case BinaryOpType.MUL:
                return l * r 
            case BinaryOpType.DIV:
                return l / r

        raise ValueError(f"Unknown OpType: {self.optype}")
