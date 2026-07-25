import numpy as np
from algebra.space import Space, ShapeTransform
from algebra.operator import ArrayOperator
from algebra.expression import ConstExpression

space = Space((10, 10))


def test_array_operator_from_sacalr():
    twos = 2.0 * np.ones(space.shape)
