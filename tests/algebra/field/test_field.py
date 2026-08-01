import numpy as np
from algebra.field import Field, stack, to_operator
from algebra.space import Space, Shape
from algebra.expression import CallableExpression
from tools.buffer import ConstValueBuffer


def test_field_component():
    space = Space((10, 10))
    shape = Shape(space, (2,))
    ones_arr = np.ones((2, 10, 10), dtype=float)
    ones_arr[1] *= 2.0
    ones = ConstValueBuffer(ones_arr)
    field = Field(shape, ones)
    assert field.component(0).shape.fieldshape() == space.shape
    assert np.allclose(field.component(0).value().eval(), 1.0)
    assert np.allclose(field.component(1).value().eval(), 2.0)

    zero = CallableExpression(shape, lambda: np.zeros(shape.fieldshape()))
    field.set_value(zero).perform()

    assert np.allclose(field.component(0).value().eval(), 0.0)
    assert np.allclose(field.component(1).value().eval(), 0.0)


def test_stack_fields():
    space = Space((10, 10))
    shape = Shape(space, ())
    ones = Field(shape, ConstValueBuffer(np.ones(space.shape)))
    twos = Field(shape, ConstValueBuffer(np.ones(space.shape) * 2.0))
    stacked = stack((ones, twos))

    assert stacked.shape.fieldshape() == (2, 10, 10)
    assert np.allclose(stacked.value().eval()[0], 1.0)
    assert np.allclose(stacked.value().eval()[1], 2.0)


def test_to_operator():
    space = Space((10, 10))
    shape = Shape(space, ())
    field_values = np.random.uniform(0, 10, space.shape)
    field = Field(shape, ConstValueBuffer(field_values))
    operator = to_operator(field)

    other_values = np.random.uniform(0, 10, space.shape)

    result = operator.apply_to(other_values)
    np.testing.assert_allclose(result, field_values * other_values)
