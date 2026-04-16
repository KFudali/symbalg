import numpy as np
from algebra.core.expression import CallableExpression
from algebra.symbolic import SymbolicExpression

def test_symbolic_expression():
    shape=(10,)
    def return_ones():
        return np.ones(shape=shape, dtype=float)
    def return_fives():
        return 5 * np.ones(shape=shape, dtype=float)
    ones = CallableExpression(return_ones, shape)
    fives = CallableExpression(return_fives, shape)

    expr = SymbolicExpression(ones)
    result = expr + fives
    expected = 6 * np.ones(shape=shape, dtype=float)
    assert np.array_equal(result.eval(), expected)