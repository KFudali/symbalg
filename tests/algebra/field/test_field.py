import numpy as np
from algebra.field import Field, stack
from algebra.space import Space, FieldShape
from algebra.expression import CallableExpression
from tools.buffer import ConstValueBuffer


def test_field_component():
    space = Space((10, 10))
    fieldshape = FieldShape(space, (2,))
    ones_arr = np.ones((2, 10, 10), dtype=float)
    ones_arr[1] *= 2.0
    ones = ConstValueBuffer(ones_arr)
    field = Field(fieldshape, ones)
    assert field.component(0).shape == space.shape
    assert np.allclose(field.component(0).value().eval(), 1.0)
    assert np.allclose(field.component(1).value().eval(), 2.0)

    zero = CallableExpression(fieldshape.shape, lambda: np.zeros(fieldshape.shape))
    field.set_value(zero).perform()

    assert np.allclose(field.component(0).value().eval(), 0.0)
    assert np.allclose(field.component(1).value().eval(), 0.0)


def test_stack_fields():
    space = Space((10, 10))
    fieldshape = FieldShape(space, ())
    ones = Field(fieldshape, ConstValueBuffer(np.ones(space.shape)))
    twos = Field(fieldshape, ConstValueBuffer(np.ones(space.shape) * 2.0))
    stacked = stack((ones, twos))

    assert stacked.shape == (2, 10, 10)
    assert np.allclose(stacked.value().eval()[0], 1.0)
    assert np.allclose(stacked.value().eval()[1], 2.0)
