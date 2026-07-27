from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from algebra.expression import Expression, SparseExpression
from algebra.space import FieldShape, SparseShape, Space
from tools.symbolic import nodes


@dataclass(frozen=True)
class ExpressionNode(nodes.ValueNode[Expression]):
    @property
    def fieldshape(self) -> FieldShape:
        return self.value.fieldshape

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def space(self) -> Space:
        return self.value.space

    def resolve(self) -> np.ndarray:
        return self.value.eval()


@dataclass(frozen=True)
class TensorOpNode(nodes.SymbolicNode[Expression]):
    left: nodes.SymbolicNode[Expression]
    right: nodes.SymbolicNode[Expression]
    subscripts: str

    def resolve(self) -> np.ndarray:
        l = self.left.resolve()
        r = self.right.resolve()
        return np.einsum(self.subscripts, l, r)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class TensorUnaryOpNode(nodes.SymbolicNode[Expression]):
    operand: nodes.SymbolicNode[Expression]
    subscripts: str

    def resolve(self) -> np.ndarray:
        a = self.operand.resolve()
        return np.einsum(self.subscripts, a)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class SparseExpressionNode(nodes.ValueNode[SparseExpression]):
    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def sparseshape(self) -> SparseShape:
        return self.value.sparseshape

    def resolve(self) -> sp.spmatrix:
        return self.value.eval()
