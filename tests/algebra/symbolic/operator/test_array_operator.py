import pytest
import numpy as np
from algebra.symbolic.array_operator import ArrayOperator
from algebra.expression import ConstExpression
from algebra.space import Space, ShapeTransform


@pytest.fixture
def ones_expr():
    s = Space((10, 10))
    return ConstExpression(s, 1.0)


@pytest.fixture
def array_lap(ones_expr):
    return ArrayOperator(ones_expr.space, ShapeTransform.NONE, ones_expr)


@pytest.fixture
def array_grad(ones_expr):
    return ArrayOperator(ones_expr.space, ShapeTransform.INCREASE_RANK, ones_expr)


@pytest.fixture
def array_div(ones_expr):
    return ArrayOperator(ones_expr.space, ShapeTransform.REDUCE_RANK, ones_expr)


def test_lap_shapes():
    space = Space((10,))
    op = ArrayOperator(space, ShapeTransform.NONE, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((10,)))
    assert out.shape == (10,)

    space = Space((10, 10))
    op = ArrayOperator(space, ShapeTransform.NONE, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((10, 10)))
    assert out.shape == (10, 10)

    out = op.apply_to(np.ones((3, 3, 10, 10)))
    assert out.shape == (3, 3, 10, 10)

    out = op.apply_to(np.ones((1, 10, 10)))
    assert out.shape == (1, 10, 10)


def test_grad_shapes():
    space = Space((10,))
    op = ArrayOperator(space, ShapeTransform.INCREASE_RANK, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((10,)))
    assert out.shape == (1, 10)

    space = Space((10, 10))
    op = ArrayOperator(space, ShapeTransform.INCREASE_RANK, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((10, 10)))
    assert out.shape == (2, 10, 10)

    out = op.apply_to(np.ones((3, 3, 10, 10)))
    assert out.shape == (3, 3, 2, 10, 10)

    out = op.apply_to(np.ones((1, 10, 10)))
    assert out.shape == (1, 2, 10, 10)


def test_div_shapes():
    space = Space((10,))
    op = ArrayOperator(space, ShapeTransform.REDUCE_RANK, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((1, 10)))
    assert out.shape == (10,)

    space = Space((10, 10))
    op = ArrayOperator(space, ShapeTransform.REDUCE_RANK, ConstExpression(space, 1.0))
    out = op.apply_to(np.ones((2, 10, 10)))
    assert out.shape == (10, 10)

    out = op.apply_to(np.ones((3, 2, 10, 10)))
    assert out.shape == (3, 10, 10)

    with pytest.raises(AssertionError):
        op.apply_to(np.ones((3, 10, 10)))


def test_lap_values(array_lap):
    inp = np.ones((10, 10), dtype=float)
    out = array_lap.apply_to(inp)
    assert np.allclose(out, 2.0)

    inp = np.ones((3, 3, 10, 10), dtype=float)
    out = array_lap.apply_to(inp)
    assert np.allclose(out, 2.0)


def test_grad_values(array_grad):
    inp = np.ones((10, 10), dtype=float)
    out = array_grad.apply_to(inp)
    assert out.shape == (2, 10, 10)
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], 1.0)

    inp = np.ones((3, 3, 10, 10), dtype=float)
    out = array_grad.apply_to(inp)
    assert out.shape == (3, 3, 2, 10, 10)
    assert np.allclose(out[:, :, 0], 1.0)
    assert np.allclose(out[:, :, 1], 1.0)


def test_div_values(array_div):
    inp = np.ones((2, 10, 10), dtype=float)
    out = array_div.apply_to(inp)
    assert np.allclose(out, 2.0)

    inp_xy = np.stack([np.ones((10, 10)) * 2.0, np.ones((10, 10)) * 3.0], axis=0)
    out = array_div.apply_to(inp_xy)
    assert np.allclose(out, 5.0)

    inp = np.ones((3, 2, 10, 10), dtype=float)
    out = array_div.apply_to(inp)
    assert out.shape == (3, 10, 10)
    assert np.allclose(out, 2.0)


def test_magics_value(array_lap):
    other = ArrayOperator(
        array_lap.space, ShapeTransform.NONE, ConstExpression(array_lap.space, 2.0)
    )

    inp = np.ones((10, 10), dtype=float)

    out = (array_lap + other).apply_to(inp)
    assert np.allclose(out, 6.0)

    out = (array_lap - other).apply_to(inp)
    assert np.allclose(out, -2.0)

    out = (array_lap * 3.0).apply_to(inp)
    assert np.allclose(out, 6.0)

    out = (-array_lap).apply_to(inp)
    assert np.allclose(out, -2.0)
