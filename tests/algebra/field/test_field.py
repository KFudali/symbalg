from algebra.field import Field
from algebra.space import Space, FieldShape
from tools.buffer import DequeValueBuffer


def test_field_component():
    space = Space((10, 10))
    fieldshape = FieldShape(space, (3,))
    field = Field(fieldshape, DequeValueBuffer(fieldshape.shape))
    assert field.component(0).shape == space.shape
