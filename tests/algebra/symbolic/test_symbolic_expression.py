import pytest
import numpy as np
from algebra.core.expression import CallableExpression, Expression
from algebra.symbolic import SymbolicExpression
from algebra.fieldshape import FieldShape

SHAPE = FieldShape((10,), 1)


@pytest.fixture
def scalar():
    from algebra.core.expression import ScalarExpression

    return ScalarExpression(2.0)


@pytest.fixture
def ones():
    def return_ones():
        return np.ones(shape=SHAPE, dtype=float)

    return CallableExpression(return_ones, SHAPE)


@pytest.fixture
def fives():
    def return_fives():
        return 5 * np.ones(shape=SHAPE, dtype=float)

    return CallableExpression(return_fives, SHAPE)


def assert_eval(expression: Expression, value: float):
    result = expression.eval()
    assert np.array_equal(result, value * np.ones_like(result))


def test_symbolic_expression_with_floats(ones):
    sym_ones = SymbolicExpression(ones)

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
    sym_ones = SymbolicExpression(ones)
    sym_fives = SymbolicExpression(fives)
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
    sym_ones = SymbolicExpression(ones)
    sym_fives = SymbolicExpression(fives)
    from algebra.core.expression import ScalarExpression

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
    sym_ones = SymbolicExpression(ones)
    sym_fives = SymbolicExpression(fives)

    add = sym_ones + sym_fives
    assert_eval(add, 6.0)

    sub = sym_fives - sym_ones
    assert_eval(sub, 4.0)

    mul = sym_ones * sym_fives
    assert_eval(mul, 5.0)

    combined = (sym_ones + sym_fives) * sym_fives - sym_ones
    assert_eval(combined, 29.0)


def test_symbolic_expression_is_immutable_on_combinations(ones, fives):
    sym_ones = SymbolicExpression(ones)
    sym_fives = SymbolicExpression(fives)
    add = sym_ones + sym_fives
    sub = sym_fives - sym_ones
    mul = sym_ones * sym_fives
    assert_eval(sym_ones, 1.0)
    assert_eval(sym_fives, 5.0)
