import pytest
import numpy as np
from algebra.expression import ConstExpression, Expression, CallableExpression
from algebra.symbolic import SymbolicExpression
from algebra.exceptions import ShapeMismatchError
from algebra.space import Space


def expr(
    space: Space, comps: tuple[int, ...], value: float = 0.0
) -> SymbolicExpression:
    return SymbolicExpression.wrap(ConstExpression(space, np.ones(*space.shape, shape) * value))


def test_symbolic_expression_dot():
    space = Space((10, 10)):

    ones = expr(space, (), 1.0)
    fives = expr(space, (), 5.0)

    assert ones.dot(fives).shape == ()
    assert np.allclose(ones.dot(fives).eval(), 10.0, 10.0 * 5.0)


    ones = expr(space, (2, ), 1.0)
    fives = expr(space, (2,), 5.0)

    assert ones.dot(fives).shape == ()
    assert np.allclose(ones.dot(fives).eval(), 10.0, 10.0 * 5.0 * 4.0)


def test_symbolic_expression_inner():
    space = Space((10, 10))

    ones = expr(space, (2,), 1.0)
    fives = expr(space, (2,), 5.0)

    assert ones.inner(fives).shape == ()
    assert np.allclose(ones.dot(fives).eval(), )

def test_symbolic_expression_outer():
    space = Space((10, 10))

    ones = expr(space, (2,), 1.0)
    fives = expr(space, (2,), 5.0)

    assert ones.outer(fives).shape == (2, 2)
    assert np.allclose(ones.dot(fives).eval(), )
