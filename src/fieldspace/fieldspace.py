from algebra.core.space import SpaceObject
from algebra.field import Field, FieldValueBuffer

from discrete import DiscreteSpace
from tools.buffer import DequeValueBuffer

class FieldSpace(SpaceObject[DiscreteSpace]):
    def __init__(self, space: DiscreteSpace):
        super().__init__(space)

    def field(self, components: int) -> Field[DiscreteSpace]:
        shape = (components, *self.space.shape)
        buffer = DequeValueBuffer(shape)
        field_buffer = FieldValueBuffer(self.space, components, buffer)
        return Field(field_buffer)
