import numpy as np
from .abstract_field import AbstractField
from .fieldshaped import FieldShaped
from algebra.expression import SimpleExpression

class FieldValue(FieldShaped, SimpleExpression):
    def __init__(self, field: AbstractField):
        FieldShaped.__init__(self, field.space, field.components)
        SimpleExpression.__init__(self, field.shape)
        self._field = field

    def copy(self) -> "FieldValue":
        return FieldValue(self._field)

    def eval(self) -> np.ndarray:
        return self._field.get_current()
