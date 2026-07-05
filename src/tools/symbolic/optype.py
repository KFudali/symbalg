from enum import Enum, auto


class OpType(Enum):
    pass


class UnaryOpType(OpType):
    NEG = auto()


class BinaryOpType(OpType):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


class MatOpType(OpType):
    DOT = auto()
    INNER = auto()
    OUTER = auto()
    TRACE = auto()


BINARY_OPS = {
    BinaryOpType.ADD: lambda a, b: a + b,
    BinaryOpType.SUB: lambda a, b: a - b,
    BinaryOpType.MUL: lambda a, b: a * b,
    BinaryOpType.DIV: lambda a, b: a / b,
}
