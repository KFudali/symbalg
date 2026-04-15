from enum import Enum, auto

class UnaryOpType(Enum):
    NEG = auto()

class BinaryOpType(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
