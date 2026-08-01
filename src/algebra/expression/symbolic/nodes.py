from dataclasses import dataclass
import numpy as np
from algebra.expression.expression import Expression, Array
from algebra.space import Space, Shape
from tools.symbolic import nodes


@dataclass(frozen=True)
class ExpressionNode(nodes.ValueNode[Expression]):
    @property
    def shape(self) -> Shape:
        return self.value.shape

    @property
    def space(self) -> Space:
        return self.value.space

    def resolve(self) -> Array:
        return self.value.eval()


@dataclass(frozen=True)
class TensorOpNode(nodes.SymbolicNode[Expression]):
    left: nodes.SymbolicNode[Expression]
    right: nodes.SymbolicNode[Expression]
    subscripts: str

    def resolve(self) -> Array:
        l = self.left.resolve()
        r = self.right.resolve()
        return np.einsum(self.subscripts, l, r)  # ignode[arg-error]


@dataclass(frozen=True)
class TensorUnaryOpNode(nodes.SymbolicNode[Expression]):
    operand: nodes.SymbolicNode[Expression]
    subscripts: str

    def resolve(self) -> Array:
        a = self.operand.resolve()
        return np.einsum(self.subscripts, a)  # type: ignore[no-untyped-call]
