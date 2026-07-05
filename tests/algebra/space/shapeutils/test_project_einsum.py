import pytest
import numpy as np

from algebra.space import utils
from algebra.space import Space, FieldShape
from tools.symbolic import MatOpType

SPACE_SHAPES = [(10,), (10, 10), (10, 10, 10)]


def _arange(shape):
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


class TestDot:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_dot_scalar_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_dot_scalar_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        a = np.ones(left.shape)
        b = 2.0 * np.ones(right.shape)
        result = np.einsum(subs, a, b)
        assert np.allclose(result, 2.0)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_dot_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_vector_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "a...,a...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_vector_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a * b, axis=0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_vector_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4,))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "ab...,b...->a..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_vector_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4,))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a * b[None, ...], axis=1)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_matrix_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3, 4))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "a...,ab...->b..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_matrix_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3, 4))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a[:, None, ...] * b, axis=0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_matrix_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4, 5))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        assert subs == "ab...,bc...->ac..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_matrix_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4, 5))
        subs = utils.project_einsum(left, right, MatOpType.DOT)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a[:, :, None, ...] * b[None, :, :, ...], axis=1)
        assert np.allclose(result, expected)


class TestInner:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_inner_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_inner_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_inner_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_inner_vector_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        assert subs == "a...,a...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_inner_vector_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a * b, axis=0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_inner_matrix_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (3, 4))
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        assert subs == "ab...,ab...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_inner_matrix_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (3, 4))
        subs = utils.project_einsum(left, right, MatOpType.INNER)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = np.sum(a * b, axis=(0, 1))
        assert np.allclose(result, expected)


class TestOuter:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_outer_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_outer_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "...,...->..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_vector_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4,))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "a...,b...->ab..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_vector_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4,))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = a[:, None, ...] * b[None, :, ...]
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_outer_vector_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (5,))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "ab...,c...->abc..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_outer_vector_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (5,))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = a[:, :, None, ...] * b[None, None, :, ...]
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_matrix_subscripts(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4, 5))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        assert subs == "a...,bc...->abc..."

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_matrix_computation(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4, 5))
        subs = utils.project_einsum(left, right, MatOpType.OUTER)
        a = _arange(left.shape)
        b = _arange(right.shape)
        result = np.einsum(subs, a, b)
        expected = a[:, None, None, ...] * b[None, :, :, ...]
        assert np.allclose(result, expected)
