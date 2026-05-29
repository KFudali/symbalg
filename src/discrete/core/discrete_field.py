from algebra.field import Field, FieldOperator
from tools.buffer import ValueBuffer
from .discrete_space import DiscreteSpace


class DiscreteField(Field):
    def __init__(self, space: DiscreteSpace, value_buffer: ValueBuffer):
        super().__init__(value_buffer, space.ndim)
        self._space = space

    @property
    def space(self) -> DiscreteSpace:
        return self._space


class DiscreteFieldOperator(FieldOperator):
    def __init__(self, field: DiscreteField, operator, expression):
        super().__init__(field, operator, expression)

    @property
    def field(self) -> DiscreteField:
        return super().field
