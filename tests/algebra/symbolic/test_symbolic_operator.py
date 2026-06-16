import numpy as np
import pytest
from algebra.expression import CallableExpression, ScalarExpression
from algebra.exceptions import ShapeMismatchError
from algebra.space import Space, ShapeTransform
from algebra.symbolic import SymbolicOperator
from conftest import MockOperator


def test_symbolic_operator_with_operator():
    op_a = MockOperator("A")
    op_b = MockOperator("B")
    op_c = MockOperator("C")
    symbolic = SymbolicOperator[MockOperator].wrap(op_a)
    result = ((symbolic + op_b) * op_c).resolve()
    assert result.name == "[[A + B] * C]"


def test_symbolic_operator_with_floats():
    op_a = MockOperator("A")
    symbolic = SymbolicOperator[MockOperator].wrap((op_a))
    result = (symbolic * 10.0).resolve()
    assert result.name == "[A * 10.0]"

    result = (symbolic / 10.0).resolve()
    assert result.name == "[A * 0.1]"

    with pytest.raises(TypeError):
        result = symbolic + 10.0

    with pytest.raises(TypeError):
        result = symbolic - 10.0

def test_symbolic_operator_with_scalar_expression():
    op_a = MockOperator("A")
    symbolic = SymbolicOperator[MockOperator].wrap(op_a)
    exp = ScalarExpression(10.0)

    result = (symbolic * exp).resolve()
    assert result.name == "[A * 10.0]"

    result = (symbolic / exp).resolve()
    assert result.name == "[A * 0.1]"

    with pytest.raises(TypeError):
        symbolic + exp

    with pytest.raises(TypeError):
        symbolic - exp


def test_symbolic_operator_shape_mismatch_raises():
    op_a = MockOperator("A")
    symbolic = SymbolicOperator[MockOperator].wrap(op_a)

    op_other_space = MockOperator("B")
    op_other_space._space = Space((20, 20))
    with pytest.raises(ShapeMismatchError):
        symbolic + op_other_space

    op_other_st = MockOperator("C")
    op_other_st._shape_transform = ShapeTransform.INCREASE_RANK
    with pytest.raises(ShapeMismatchError):
        symbolic + op_other_st

    non_scalar_expr = CallableExpression((10,), lambda: np.ones(10))
    with pytest.raises(ShapeMismatchError):
        symbolic * non_scalar_expr
