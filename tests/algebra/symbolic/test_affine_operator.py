import pytest
import numpy as np


from algebra.symbolic import AffineOperator
from algebra.core.expression import CallableExpression
from conftest import MockOperator


@pytest.fixture
def double_add_one():
    double_op = MockOperator("A")

    def double(input_field, output_field):
        output_field[:] = 2 * input_field[:]

    double_op.set_apply(double)

    def return_ones():
        return np.ones(shape=(10,), dtype=float)

    ones = CallableExpression(return_ones, (10,))
    operator = AffineOperator(double_op, ones)
    return operator


@pytest.fixture
def triple_sub_three():
    tiple_op = MockOperator("B")

    def triple(input_field, output_field):
        output_field[:] = 3 * input_field[:]

    tiple_op.set_apply(triple)

    def return_threes():
        return -3.0 * np.ones(shape=(10,), dtype=float)

    threes = CallableExpression(return_threes, (10,))
    operator = AffineOperator(tiple_op, threes)
    return operator


def test_affine_operator_double_apply(double_add_one):
    input_field = np.ones(shape=(10,), dtype=float)
    output_field = np.zeros(shape=(10,), dtype=float)

    double_add_one.apply(input_field, output_field)
    assert np.allclose(output_field, 3.0)

    input_field = 5 * np.ones(shape=(10,), dtype=float)
    output_field = np.zeros(shape=(10,), dtype=float)

    double_add_one.apply(input_field, output_field)
    assert np.allclose(output_field, 11.0)


def test_affine_operator_with_affine(double_add_one, triple_sub_three):
    op_add = double_add_one + triple_sub_three
    assert op_add.operator.resolve().name == "[A + B]"
    assert np.allclose(op_add.expression.resolve(), -2.0)

    op_sub = double_add_one - triple_sub_three
    assert op_sub.operator.resolve().name == "[A + [-B]]"
    assert np.allclose(op_sub.expression.resolve(), 4.0)

    op_mul = double_add_one * triple_sub_three
    assert op_mul.operator.resolve().name == "[A * B]"
    assert np.allclose(op_mul.expression.resolve(), -3.0)

    op_div = double_add_one / triple_sub_three
    assert op_div.operator.resolve().name == "[A / B]"
    assert np.allclose(op_div.expression.resolve(), (1.0 / (-3.0)))

    op_combined = double_add_one + triple_sub_three
    op_combined *= triple_sub_three
    assert op_combined.operator.resolve().name == "[[A + B] * B]"
    assert np.allclose(op_combined.expression.resolve(), (-2.0 * (-3.0)))


def test_affine_operator_with_floats(double_add_one):
    op_add = double_add_one + 1.0
    assert op_add.operator.resolve().name == "A"
    assert np.allclose(op_add.expression.resolve(), 2.0)

    op_sub = double_add_one - 1.0
    assert op_sub.operator.resolve().name == "A"
    assert np.allclose(op_sub.expression.resolve(), 0.0)

    op_mul = double_add_one * 3.0
    assert op_mul.operator.resolve().name == "[A * 3.0]"
    assert np.allclose(op_mul.expression.resolve(), 3.0)

    op_div = double_add_one / 2.0
    assert op_div.operator.resolve().name == "[A * 0.5]"
    assert np.allclose(op_div.expression.resolve(), 0.5)


def test_affine_operator_with_scalar_exp(double_add_one):
    from algebra.core.expression import ScalarExpression

    scalar = ScalarExpression(2.0)

    op_add = double_add_one + scalar
    assert op_add.operator.resolve().name == "A"
    assert np.allclose(op_add.expression.resolve(), 3.0)

    op_sub = double_add_one - scalar
    assert op_sub.operator.resolve().name == "A"
    assert np.allclose(op_sub.expression.resolve(), -1.0)

    op_mul = double_add_one * scalar
    assert op_mul.operator.resolve().name == "[A * 2.0]"
    assert np.allclose(op_mul.expression.resolve(), 2.0)

    op_div = double_add_one / scalar
    assert op_div.operator.resolve().name == "[A * 0.5]"
    assert np.allclose(op_div.expression.resolve(), 0.5)


def test_affine_operator_with_expression(double_add_one):
    two_arr = 2.0 * np.ones(shape=(10,), dtype=float)

    def return_twos():
        return two_arr

    twos = CallableExpression(return_twos, (10,))

    op_add = double_add_one + twos
    assert op_add.operator.resolve().name == "A"
    assert np.allclose(op_add.expression.resolve(), 3.0)

    op_sub = double_add_one - twos
    assert op_sub.operator.resolve().name == "A"
    assert np.allclose(op_sub.expression.resolve(), -1.0)

    op_mul = double_add_one * twos
    assert op_mul.operator.resolve().name == f"[A * {two_arr}]"
    assert np.allclose(op_mul.expression.resolve(), 2.0)

    op_div = double_add_one / twos
    assert op_div.operator.resolve().name == f"[A * {1.0 / two_arr}]"
    assert np.allclose(op_div.expression.resolve(), 0.5)


def test_affine_operator_with_operator(double_add_one):
    op = MockOperator("B")

    op_add = double_add_one + op
    assert op_add.operator.resolve().name == "[A + B]"
    assert np.allclose(op_add.expression.resolve(), 1.0)

    op_sub = double_add_one - op
    assert op_sub.operator.resolve().name == "[A + [-B]]"
    assert np.allclose(op_sub.expression.resolve(), 1.0)

    op_mul = double_add_one * op
    assert op_mul.operator.resolve().name == "[A * B]"
    assert np.allclose(op_mul.expression.resolve(), 1.0)

    op_div = double_add_one / op
    assert op_div.operator.resolve().name == "[A / B]"
    assert np.allclose(op_div.expression.resolve(), 1.0)
