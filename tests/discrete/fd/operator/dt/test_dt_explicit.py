import numpy as np
from discrete.fd.operator.dt import explicit
from algebra.expression import CallableExpression
from algebra.space import Shape, Space
from algebra.field import Field
from tools.buffer import DequeValueBuffer

space = Space((10, 10))


def field_buffer() -> tuple[Field, DequeValueBuffer]:
    shape = Shape(space, (1,))
    buffer = DequeValueBuffer(shape.fieldshape())
    return Field(shape, buffer), buffer


def test_first_order_with_const_field():
    time_step = CallableExpression(Shape.scalar(space), lambda: np.array(0.01))
    field, values = field_buffer()
    values.set(np.ones(shape=field.shape.fieldshape(), dtype=float))
    dt = explicit.bfd(field, time_step, order=1)
    for _ in range(10):
        values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float))
        vals = dt.apply_to(field.value().eval())
        assert np.allclose(vals, 0.0)


def test_second_order_with_const_field():
    time_step = CallableExpression(Shape.scalar(space), lambda: np.array(0.01))
    field, values = field_buffer()
    values.set_saved_steps(2)
    values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float))
    values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float))
    dt = explicit.bfd(field, time_step, order=2)
    for _ in range(10):
        values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float))
        vals = dt.apply_to(field.value().eval())
        assert np.allclose(vals, 0.0)


def test_first_order_with_linear_field():
    time_step = CallableExpression(Shape.scalar(space), lambda: np.array(0.01))
    field, values = field_buffer()
    dt = explicit.bfd(field, time_step, order=1)
    for t in range(10):
        values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float) * t)
        if t > 0:
            vals = dt.apply_to(field.value().eval())
            assert np.allclose(vals, 1.0 / 0.01)


def test_second_order_with_linear_field():
    time_step = CallableExpression(Shape.scalar(space), lambda: np.array(0.01))
    field, values = field_buffer()
    dt = explicit.bfd(field, time_step, order=2)
    for t in range(10):
        values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float) * t)
        if t > 1:
            vals = dt.apply_to(field.value().eval())
            assert np.allclose(vals, 1.0 / 0.01)


def test_second_order_with_square_field():
    dt_value = 0.01
    time_step = CallableExpression(Shape.scalar(space), lambda: np.array(dt_value))
    field, values = field_buffer()
    dt = explicit.bfd(field, time_step, order=2)
    for t in range(10):
        time = t * dt_value
        values.advance(np.ones(shape=field.shape.fieldshape(), dtype=float) * time**2)
        if t > 1:
            vals = dt.apply_to(field.value().eval())
            assert np.allclose(vals, 2.0 * time)
