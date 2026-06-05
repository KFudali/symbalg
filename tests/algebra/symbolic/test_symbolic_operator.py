import pytest
from algebra.expression import ScalarExpression
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
