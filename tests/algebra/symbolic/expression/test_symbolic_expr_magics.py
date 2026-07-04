import pytest
import numpy as np
from algebra.expression import ConstExpression, Expression, CallableExpression
from algebra.symbolic import SymbolicExpression
from algebra.exceptions import ShapeMismatchError
from algebra.space import Space

SHAPE = (10,)


@pytest.fixture
def ones() -> SymbolicExpression:
    space = Space(SHAPE)
    return SymbolicExpression.wrap(ConstExpression(space, np.ones(SHAPE)))


@pytest.fixture
def fives() -> SymbolicExpression:
    space = Space(SHAPE)
    return SymbolicExpression.wrap(ConstExpression(space, np.ones(SHAPE) * 5.0))


def assert_eval(expression: Expression, value: float):
    result = expression.eval()
    assert np.array_equal(result, value * np.ones_like(result))


def test_symbolic_expression_with_floats(ones: SymbolicExpression):
    add = ones + 1.0
    assert_eval(add, 2.0)
    add += 3.0
    assert_eval(add, 5.0)
    add += 5.0
    assert_eval(add, 10.0)

    sub = ones - 1.0
    assert_eval(sub, 0.0)
    sub -= 5.0
    assert_eval(sub, -5.0)
    sub -= -10.0
    assert_eval(sub, 5.0)

    mul = ones * 3.0
    assert_eval(mul, 3.0)
    mul *= 3.0
    assert_eval(mul, 9.0)
    mul *= 5.0
    assert_eval(mul, 45.0)

    eq = (((ones * 3.0) + 7.0) / 5.0) - 5.0
    assert_eval(eq, -3.0)


def test_symbolic_expression_with_arrays(
    ones: SymbolicExpression, fives: SymbolicExpression
):

    array = np.ones(shape=SHAPE, dtype=float) * 3.0

    add = ones + array
    assert_eval(add, 4.0)
    add += array
    assert_eval(add, 7.0)

    sub = fives - array
    assert_eval(sub, 2.0)
    sub -= array
    assert_eval(sub, -1.0)

    mul = ones * array
    assert_eval(mul, 3.0)
    mul *= array
    assert_eval(mul, 9.0)


def test_symbolic_expression_with_scalar_expression(
    ones: SymbolicExpression, fives: SymbolicExpression
):
    scalar = ConstExpression(ones.space, 2.0)

    add = ones + scalar
    assert_eval(add, 3.0)
    add += scalar
    assert_eval(add, 5.0)

    sub = fives - scalar
    assert_eval(sub, 3.0)
    sub -= scalar
    assert_eval(sub, 1.0)

    mul = ones * scalar
    assert_eval(mul, 2.0)
    mul *= scalar
    assert_eval(mul, 4.0)

    div = fives / scalar
    assert_eval(div, 2.5)
    div /= scalar
    assert_eval(div, 1.25)


def test_symbolic_expression_with_expression(
    ones: SymbolicExpression, fives: SymbolicExpression
):
    add = ones + fives
    assert_eval(add, 6.0)

    sub = fives - ones
    assert_eval(sub, 4.0)

    mul = ones * fives
    assert_eval(mul, 5.0)

    combined = (ones + fives) * fives - ones
    assert_eval(combined, 29.0)


def test_symbolic_expression_shape_mismatch_raises(ones: SymbolicExpression):
    ones = SymbolicExpression.wrap(ones)
    other = ConstExpression(ones.space, np.ones(shape=(2, *SHAPE)))

    with pytest.raises(ShapeMismatchError):
        ones + other

    with pytest.raises(ShapeMismatchError):
        ones - other

    with pytest.raises(ShapeMismatchError):
        ones * other

    with pytest.raises(ShapeMismatchError):
        ones / other


def test_symbolic_expression_combination_broadcasts_shape(ones: SymbolicExpression):
    """Combining a scalar-shape SymbolicExpression with an array-shape one
    must yield a SymbolicExpression whose ``shape`` matches the array
    operand, regardless of which side initiates the operation.
    """
    scalar = SymbolicExpression.wrap(ConstExpression(ones.space, 3.0))

    assert scalar.shape == ()
    assert ones.shape == SHAPE

    # scalar * array
    combined = scalar * ones
    assert combined.shape == SHAPE
    assert combined.eval().shape == SHAPE

    # array * scalar (rmul / mul from the array side)
    combined = ones * scalar
    assert combined.shape == SHAPE
    assert combined.eval().shape == SHAPE

    # float * array (via __rmul__)
    combined = 2.0 * ones
    assert combined.shape == SHAPE

    # scalar + array
    combined = scalar + ones
    assert combined.shape == SHAPE
    assert combined.eval().shape == SHAPE


def test_symbolic_expression_is_immutable_on_combinations(
    ones: SymbolicExpression, fives: SymbolicExpression
):
    add = ones + fives
    sub = fives - ones
    mul = ones * fives
    assert_eval(ones, 1.0)
    assert_eval(fives, 5.0)
