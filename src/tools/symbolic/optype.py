from enum import Enum, auto

class UnaryOpType(Enum):
    NEG = auto()

class BinaryOpType(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

BINARY_OPS = {
    BinaryOpType.ADD: lambda a, b: a + b,
    BinaryOpType.SUB: lambda a, b: a - b,
    BinaryOpType.MUL: lambda a, b: a * b,
    BinaryOpType.DIV: lambda a, b: a / b
}
