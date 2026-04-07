from typing import Self
import numpy as np
from .expression import Expression

class ExpressionWrapper(Expression):
    def __init__(self, expression: Expression):
        super().__init__(expression.output_shape)
        self._expression = expression

    @property
    def expression(self) -> Expression:
        return self._expression

    def eval(self) -> np.ndarray:
        return self._expression.eval()
    
    def copy(self) -> Self:
        return ExpressionWrapper(self._expression)
