import pytest
import numpy as np
from algebra.expression import CallableExpression, ScalarExpression, Expression
from algebra.symbolic import SymbolicExpression
from algebra.exceptions import ShapeMismatchError

SHAPE = (10,)


@pytest.fixture
def scalar():
    return ScalarExpression(2.0)


@pytest.fixture
def ones():
    def return_ones():
        return np.ones(shape=SHAPE, dtype=float)

    return CallableExpression(SHAPE, return_ones)


@pytest.fixture
def fives():
    def return_fives():
        return 5 * np.ones(shape=SHAPE, dtype=float)

    return CallableExpression(SHAPE, return_fives)


def assert_eval(expression: Expression, value: float):
    result = expression.eval()
    assert np.array_equal(result, value * np.ones_like(result))


def test_symbolic_expression_with_floats(ones):
    sym_ones = SymbolicExpression.wrap(ones)

    add = sym_ones + 1.0
    assert_eval(add, 2.0)
    add += 3.0
    assert_eval(add, 5.0)
    add += 5.0
    assert_eval(add, 10.0)

    sub = sym_ones - 1.0
    assert_eval(sub, 0.0)
    sub -= 5.0
    assert_eval(sub, -5.0)
    sub -= -10.0
    assert_eval(sub, 5.0)

    mul = sym_ones * 3.0
    assert_eval(mul, 3.0)
    mul *= 3.0
    assert_eval(mul, 9.0)
    mul *= 5.0
    assert_eval(mul, 45.0)

    eq = (((sym_ones * 3.0) + 7.0) / 5.0) - 5.0
    assert_eval(eq, -3.0)


def test_symbolic_expression_with_arrays(ones, fives):
    sym_ones = SymbolicExpression.wrap(ones)
    sym_fives = SymbolicExpression.wrap(fives)
    array = np.ones(shape=SHAPE, dtype=float) * 3.0

    add = sym_ones + array
    assert_eval(add, 4.0)
    add += array
    assert_eval(add, 7.0)

    sub = sym_fives - array
    assert_eval(sub, 2.0)
    sub -= array
    assert_eval(sub, -1.0)

    mul = sym_ones * array
    assert_eval(mul, 3.0)
    mul *= array
    assert_eval(mul, 9.0)


def test_symbolic_expression_with_scalar_expression(ones, fives):
    sym_ones = SymbolicExpression.wrap(ones)
    sym_fives = SymbolicExpression.wrap(fives)
    scalar = ScalarExpression(2.0)

    add = sym_ones + scalar
    assert_eval(add, 3.0)
    add += scalar
    assert_eval(add, 5.0)

    sub = sym_fives - scalar
    assert_eval(sub, 3.0)
    sub -= scalar
    assert_eval(sub, 1.0)

    mul = sym_ones * scalar
    assert_eval(mul, 2.0)
    mul *= scalar
    assert_eval(mul, 4.0)

    div = sym_fives / scalar
    assert_eval(div, 2.5)
    div /= scalar
    assert_eval(div, 1.25)


def test_symbolic_expression_with_expression(ones, fives):
    sym_ones = SymbolicExpression.wrap(ones)
    sym_fives = SymbolicExpression.wrap(fives)

    add = sym_ones + sym_fives
    assert_eval(add, 6.0)

    sub = sym_fives - sym_ones
    assert_eval(sub, 4.0)

    mul = sym_ones * sym_fives
    assert_eval(mul, 5.0)

    combined = (sym_ones + sym_fives) * sym_fives - sym_ones
    assert_eval(combined, 29.0)


def test_symbolic_expression_shape_mismatch_raises(ones):
    sym_ones = SymbolicExpression.wrap(ones)
    other_shape = (SHAPE[0] + 1,)
    other_array = np.ones(shape=other_shape, dtype=float)
    other_expr = CallableExpression(other_shape, lambda: other_array)

    with pytest.raises(ShapeMismatchError):
        sym_ones + other_array

    with pytest.raises(ShapeMismatchError):
        sym_ones + other_expr

    with pytest.raises(ShapeMismatchError):
        sym_ones * other_expr


def test_symbolic_expression_is_immutable_on_combinations(ones, fives):
    sym_ones = SymbolicExpression.wrap(ones)
    sym_fives = SymbolicExpression.wrap(fives)
    add = sym_ones + sym_fives
    sub = sym_fives - sym_ones
    mul = sym_ones * sym_fives
    assert_eval(sym_ones, 1.0)
    assert_eval(sym_fives, 5.0)
