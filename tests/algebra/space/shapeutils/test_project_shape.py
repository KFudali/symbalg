import pytest
import numpy as np

from algebra.space import utils
from algebra.space import Space, FieldShape
from tools.symbolic import MatOpType

SPACE_SHAPES = [(10,), (10, 10), (10, 10, 10)]


class TestDot:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_dot_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == ()
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_dot_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == ()
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4,))
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_dot_matrix(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3, 4))
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == (4,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_dot_matrix(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (4, 5))
        result = utils.project_shape(left, right, MatOpType.DOT)
        assert result.components == (3, 5)
        assert result.space is left.space


class TestInner:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_inner_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.INNER)
        assert result.components == ()
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_inner_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        result = utils.project_shape(left, right, MatOpType.INNER)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_inner_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.INNER)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_inner_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (3,))
        result = utils.project_shape(left, right, MatOpType.INNER)
        assert result.components == ()
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_inner_matrix(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (3, 4))
        result = utils.project_shape(left, right, MatOpType.INNER)
        assert result.components == ()
        assert result.space is left.space


class TestOuter:
    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_outer_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == ()
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_scalar_outer_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, ())
        right = FieldShape(space, (3,))
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_scalar(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, ())
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == (3,)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4,))
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == (3, 4)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_matrix_outer_vector(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3, 4))
        right = FieldShape(space, (5,))
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == (3, 4, 5)
        assert result.space is left.space

    @pytest.mark.parametrize("space_shape", SPACE_SHAPES)
    def test_vector_outer_matrix(self, space_shape):
        space = Space(space_shape)
        left = FieldShape(space, (3,))
        right = FieldShape(space, (4, 5))
        result = utils.project_shape(left, right, MatOpType.OUTER)
        assert result.components == (3, 4, 5)
        assert result.space is left.space
