import pytest
import numpy as np
from algebra.exceptions import ShapeMismatchError
from algebra.core.expression import ScalarExpression, CallableExpression
from algebra.symbolic import SymbolicOperator
from conftest import MockOperator

def test_symbolic_operator_with_operator():
    op_a = MockOperator("A")
    op_b = MockOperator("B")
    op_c = MockOperator("C")
    symbolic = SymbolicOperator[MockOperator](op_a)
    result = ((symbolic + op_b) * op_c).fold()
    assert result.name == "[[A + B] * C]"

def test_symbolic_operator_with_floats():
    op_a = MockOperator("A")
    symbolic = SymbolicOperator[MockOperator](op_a)
    result = (symbolic * 10.0).fold()
    assert result.name == "[A * 10.0]"

    result = (symbolic / 10.0).fold()
    assert result.name == "[A * 0.1]"

    with pytest.raises(ValueError):
        result = symbolic + 10.0

    with pytest.raises(ValueError):
        result = symbolic - 10.0

def test_symbolic_operator_with_array():
    op_a = MockOperator("A", input_shape=(2,), output_shape=(2,))
    symbolic = SymbolicOperator[MockOperator](op_a)
    arr = np.array([2.0, 2.0])
    result = (symbolic * arr).fold()
    assert result.name == f"[A * {arr}]"

    result = (symbolic / arr).fold()
    assert result.name == f"[A * {1.0 / arr}]"

    with pytest.raises(ValueError):
        result = symbolic + arr

    with pytest.raises(ValueError):
        result = symbolic - arr

    arr = np.array([2.0, 2.0, 2.0])
    with pytest.raises(ShapeMismatchError):
        result = symbolic * arr

def test_symbolic_operator_with_scalar_expression():
    op_a = MockOperator("A")
    symbolic = SymbolicOperator[MockOperator](op_a)
    exp = ScalarExpression(10.0)

    result = (symbolic * exp).fold()
    assert result.name == "[A * 10.0]"

    result = (symbolic / exp).fold()
    assert result.name == "[A * 0.1]"

    with pytest.raises(ValueError):
        result = symbolic + exp

    with pytest.raises(ValueError):
        result = symbolic - exp

def test_symbolic_operator_with_expression():
    op_a = MockOperator("A", input_shape=(2,), output_shape=(2,))
    symbolic = SymbolicOperator[MockOperator](op_a)
    twos = 2.0*np.ones(shape=(2,), dtype=float)
    def return_twos():
        return twos
    exp = CallableExpression(return_twos, (2,))

    result = (symbolic * exp).fold()
    assert result.name == f"[A * {twos}]"

    result = (symbolic / exp).fold()
    assert result.name == f"[A * {1.0 / twos}]"

    with pytest.raises(ValueError):
        result = symbolic + exp

    with pytest.raises(ValueError):
        result = symbolic - exp

    exp = CallableExpression(return_twos, (10,10))
    with pytest.raises(ShapeMismatchError):
        result = symbolic * exp

