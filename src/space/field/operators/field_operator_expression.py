import numpy as np
from algebra.operator import OperatorExpression, CombinedOperator
from ..core import FieldShaped, FieldValue

class FieldOperatorExpression(OperatorExpression[CombinedOperator], FieldShaped):
    def __init__(self, input: FieldValue, operator: CombinedOperator):
        OperatorExpression.__init__(self, input, operator)
        FieldShaped.__init__(self, input.space, input.components)

    def copy(self):
        return FieldOperatorExpression(self.input.copy(), self.operator.copy())

    def eval(self) -> np.ndarray:
        out = np.zeros(shape=(self.components, self.output_shape))
        input = self._input.eval()
        for comp in range(self.components):
            self._operator.apply(input[comp][...], out[comp][...])
        return out
