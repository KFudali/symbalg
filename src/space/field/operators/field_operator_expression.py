import numpy as np
from algebra.operator import SymbolicOperator, Operator
from algebra.expression import SymbolicExpression, Expression, ZeroExpression
from algebra.exceptions import ShapeMismatchError
from space.field.core import AbstractField

class FieldOperatorExpression(Operator, Expression):
    def __init__(
        self, 
        field: AbstractField, 
        operator: Operator, 
        lifting: Expression | None = None
    ):
        if field.shape != operator.input_shape:
            raise ShapeMismatchError("Field has to match operator.")
        if not lifting:
            lifting = ZeroExpression(operator.input_shape)
        
        self._operator_tree = SymbolicOperator(operator)
        self._lifting_tree = SymbolicExpression(lifting)
        self._field = field

    def eval(self) -> np.ndarray:
        return super().eval()
    