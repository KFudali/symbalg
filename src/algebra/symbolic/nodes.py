from dataclasses import dataclass
import numpy as np
from algebra.expression import Expression
from algebra.space import FieldShape, Space
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

    def resolve(self) -> np.ndarray | float:
        return self.value.eval()
