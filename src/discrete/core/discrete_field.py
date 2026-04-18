from algebra.field import Field, FieldValueBuffer, FieldOperator
from .discrete_space import DiscreteSpace

class DiscreteFieldValueBuffer(FieldValueBuffer[DiscreteSpace]):
    pass

class DiscreteField(Field[DiscreteSpace]):
    def __init__(self, value_buffer: DiscreteFieldValueBuffer):
        super().__init__(value_buffer)
        self.space.time.advanceables.register(self)

class DiscreteFieldOperator(FieldOperator[DiscreteSpace]):
    pass
