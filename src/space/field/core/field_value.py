import numpy as np
from .abstract_field import AbstractField
from .fieldshaped import FieldShaped
from algebra.expression import SymbolicExpression, CallableExpression

class FieldValue(SymbolicExpression, FieldShaped):
    def __init__(self, field: AbstractField):
        FieldShaped.__init__(self, field.space, field.components)
        self._field = field
        expr_call = field.get_current
        expr = CallableExpression(expr_call, self.shape)
        SymbolicExpression.__init__(expr)