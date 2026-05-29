from __future__ import annotations
import numpy as np
from .expression import Expression
from .space import FieldShaped


class Field(FieldShaped):
    def __init__(self, value_buffer):
        self._value_buffer = value_buffer

    def past(self, step: int) -> "Field":
        pass

    def value(self) -> Expression:
        return self._value_buffer.get()

    def value(self) -> Expression:
        return self._value_buffer.get()
