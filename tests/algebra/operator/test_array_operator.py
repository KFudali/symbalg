import pytest
import numpy as np
import scipy.sparse as sp
from algebra.exceptions import ShapeMismatchError
from algebra.space import Space, ShapeTransform
from algebra.operator import ArrayOperator
from algebra.expression import ConstSparseExpression

space = Space((10, 10))
N = int(np.prod(space.shape))


def test_array_operator_multiplies_pointwise():
    weights = np.random.uniform(0, 10, N)
    weights_expr = ConstSparseExpression(space, sp.diags(weights, 0))
    operator = ArrayOperator(space, ShapeTransform.NONE, weights_expr)

    field = np.random.uniform(0, 10, space.shape)
    result = operator.apply_to(field)

    assert result.shape == space.shape
    assert np.allclose(result, field * weights.reshape(space.shape))


def test_array_operator_raises_on_wrong_matrix_shape():
    twos = sp.diags(2.0 * np.ones(2 * N), 0)
    with pytest.raises(ShapeMismatchError):
        ConstSparseExpression(space, twos)


def test_array_operator_increase_rank():
    twos = sp.diags(2.0 * np.ones(N), 0)
    weights = ConstSparseExpression(space, twos)
    operator = ArrayOperator(space, ShapeTransform.INCREASE_RANK, weights)
    field = np.ones(space.shape)
    result = operator.apply_to(field)

    assert result.shape == (space.ndim, *space.shape)
    assert np.allclose(result, 2.0)


def test_array_operator_reduce_rank():
    twos = sp.diags(2.0 * np.ones(N), 0)
    weights = ConstSparseExpression(space, twos)
    operator = ArrayOperator(space, ShapeTransform.REDUCE_RANK, weights)
    field = np.ones((space.ndim, *space.shape))
    result = operator.apply_to(field)

    assert result.shape == space.shape
    assert np.allclose(result, 4.0)
