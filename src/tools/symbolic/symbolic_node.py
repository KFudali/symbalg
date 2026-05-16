from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
from .optype import UnaryOpType, BinaryOpType

TSymbolic = TypeVar("TSymbolic")


class SymbolicNode(ABC, Generic[TSymbolic]):
    @abstractmethod
    def resolve(self) -> TSymbolic:
        pass


@dataclass(frozen=True)
class ValueNode(SymbolicNode[TSymbolic]):
    value: TSymbolic

    def resolve(self) -> TSymbolic:
        return self.value


@dataclass(frozen=True)
class UnaryNode(SymbolicNode[TSymbolic]):
    optype: UnaryOpType
    operand: SymbolicNode[TSymbolic]

    def resolve(self) -> TSymbolic:
        value = self.operand.resolve()
        if self.optype == UnaryOpType.NEG:
            return -value
        raise ValueError(f"Unknown OpType: {self.optype}")


@dataclass(frozen=True)
class BinaryNode(SymbolicNode[TSymbolic]):
    optype: BinaryOpType
    left: SymbolicNode[TSymbolic]
    right: SymbolicNode[TSymbolic]

    def resolve(self) -> TSymbolic:
        l = self.left.resolve()
        r = self.right.resolve()

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
