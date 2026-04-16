import numpy as np
from algebra.core.expression import ScalarExpression

def test_scalar_expression_with_const():
    one = ScalarExpression(1)
    one_f = ScalarExpression(1.0)
    one_arr = ScalarExpression(np.array(1.0))

    for exp in [one, one_f, one_arr]:
        assert np.isclose(exp.eval(), 1.0)

def test_scalar_expression_with_getter():
    def get_one():
        return 1.0
    one = ScalarExpression(get_one)
    assert np.isclose(one.eval(), 1.0)
