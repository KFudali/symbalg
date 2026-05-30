from dataclasses import dataclass
import numpy as np
from algebra.expression import Expression, ScalarExpression
from tools.symbolic import nodes


@dataclass(frozen=True)
class ExpressionNode(nodes.ValueNode[Expression]):
    def resolve(self) -> np.ndarray:
        return self.value.eval()


@dataclass(frozen=True)
class ExprScaleNode(nodes.ValueNode[ScalarExpression]):
    def resolve(self) -> float:
        return float(self.value.eval())
