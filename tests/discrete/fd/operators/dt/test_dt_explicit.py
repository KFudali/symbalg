import numpy as np
from discrete.fd.operators.dt import explicit
from algebra.expression import ScalarExpression
from algebra.space import Space
from algebra.field import Field, FieldShape
from tools.buffer import DequeValueBuffer

space = Space((10, 10))


def field_buffer() -> tuple[Field, DequeValueBuffer]:
    fieldshape = FieldShape(space, (1,))
    buffer = DequeValueBuffer(fieldshape.shape)
    return Field(fieldshape, buffer), buffer


def test_first_order_with_const_field():
    time_step = ScalarExpression(0.01)
    field, values = field_buffer()
    values.set(np.ones(shape=field.shape, dtype=float))
    dt = explicit.bfd(field, time_step, order=1)
    for _ in range(10):
        values.advance(np.ones(shape=field.shape, dtype=float))
        vals = dt.of(field).eval()
        assert np.allclose(vals, 0.0)


def test_second_order_with_const_field():
    time_step = ScalarExpression(0.01)
    field, values = field_buffer()
    values.set_saved_steps(2)
    values.advance(np.ones(shape=field.shape, dtype=float))
    values.advance(np.ones(shape=field.shape, dtype=float))
    dt = explicit.bfd(field, time_step, order=2)
    for _ in range(10):
        values.advance(np.ones(shape=field.shape, dtype=float))
        vals = dt.of(field).eval()
        assert np.allclose(vals, 0.0)


def test_first_order_with_linear_field():
    time_step = ScalarExpression(0.01)
    field, values = field_buffer()
    dt = explicit.bfd(field, time_step, order=1)
    for t in range(10):
        values.advance(np.ones(shape=field.shape, dtype=float) * t)
        if t > 0:
            vals = dt.of(field).eval()
            assert np.allclose(vals, 1.0 / 0.01)


def test_second_order_with_linear_field():
    time_step = ScalarExpression(0.01)
    field, values = field_buffer()
    dt = explicit.bfd(field, time_step, order=2)
    for t in range(10):
        values.advance(np.ones(shape=field.shape, dtype=float) * t)
        if t > 1:
            vals = dt.of(field).eval()
            assert np.allclose(vals, 1.0 / 0.01)
