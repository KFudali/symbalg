import numpy as np
import pytest

from algebra.expression.symbolic import SymbolicExpression
from algebra.expression import ConstFieldExpression
from algebra.space import Space


def expr(
    space: Space,
    components: tuple[int, ...],
    value: float = 0.0,
) -> SymbolicExpression:
    return SymbolicExpression.wrap(
        ConstFieldExpression(
            space,
            np.ones((*components, *space.shape)) * value,
        )
    )


SPACES = [
    Space((10,)),
    Space((10, 10)),
    Space((10, 10, 10)),
]


class TestExprShape:
    @pytest.mark.parametrize("space", SPACES)
    def test_expr_shape(self, space: Space):
        A = expr(space, (2, 3))
        assert A.components == (2, 3)
        assert A.shape.fieldshape() == (2, 3, *space.shape)


class TestDot:
    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_dot_scalar(self, space: Space):
        ones = expr(space, (), 1.0)
        fives = expr(space, (), 5.0)
        result = ones.dot(fives)
        assert result.components == ()
        assert np.allclose(result.eval(), 5.0)

    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_dot_vector(self, space: Space):
        s = expr(space, ())
        v2 = expr(space, (2,))
        assert s.dot(v2).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_dot_scalar(self, space: Space):
        v2 = expr(space, (2,))
        s = expr(space, ())
        assert v2.dot(s).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_dot_vector(self, space: Space):
        ones = expr(space, (2,), 1.0)
        fives = expr(space, (2,), 5.0)
        result = ones.dot(fives)
        assert result.components == ()
        assert np.allclose(result.eval(), 10.0)

    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_dot_matrix(self, space: Space):
        s = expr(space, ())
        A = expr(space, (2, 3))
        assert s.dot(A).components == (2, 3)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_dot_scalar(self, space: Space):
        A = expr(space, (2, 3))
        s = expr(space, ())
        assert A.dot(s).components == (2, 3)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_dot_vector(self, space: Space):
        A = expr(space, (2, 3))
        v3 = expr(space, (3,))
        assert A.dot(v3).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_dot_matrix(self, space: Space):
        v3 = expr(space, (3,))
        B = expr(space, (3, 4))
        assert v3.dot(B).components == (4,)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_dot_matrix(self, space: Space):
        A = expr(space, (2, 3))
        B = expr(space, (3, 4))
        C = expr(space, (4, 5))
        assert A.dot(B).components == (2, 4)
        assert B.dot(C).components == (3, 5)

    @pytest.mark.parametrize("space", SPACES)
    def test_chained_dot(self, space: Space):
        A = expr(space, (2, 3))
        B = expr(space, (3, 4))
        C = expr(space, (4, 5))
        assert A.dot(B).dot(C).components == (2, 5)


class TestInner:
    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_inner_scalar(self, space: Space):
        s = expr(space, ())
        assert s.inner(s).components == ()

    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_inner_vector(self, space: Space):
        s = expr(space, ())
        v = expr(space, (2,))
        assert s.inner(v).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_inner_scalar(self, space: Space):
        v = expr(space, (2,))
        s = expr(space, ())
        assert v.inner(s).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_inner_vector(self, space: Space):
        ones = expr(space, (2,), 1.0)
        fives = expr(space, (2,), 5.0)
        result = ones.inner(fives)
        assert result.components == ()
        assert np.allclose(result.eval(), 10.0)

    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_inner_matrix(self, space: Space):
        s = expr(space, ())
        A = expr(space, (2, 3))
        assert s.inner(A).components == (2, 3)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_inner_scalar(self, space: Space):
        A = expr(space, (2, 3))
        s = expr(space, ())
        assert A.inner(s).components == (2, 3)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_inner_matrix(self, space: Space):
        A = expr(space, (2, 3))
        assert A.inner(A).components == ()


class TestOuter:
    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_outer_scalar(self, space: Space):
        s = expr(space, ())
        assert s.outer(s).components == ()

    @pytest.mark.parametrize("space", SPACES)
    def test_scalar_outer_vector(self, space: Space):
        s = expr(space, ())
        v = expr(space, (2,))
        assert s.outer(v).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_outer_scalar(self, space: Space):
        v = expr(space, (2,))
        s = expr(space, ())
        assert v.outer(s).components == (2,)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_outer_vector(self, space: Space):
        ones = expr(space, (2,), 1.0)
        fives = expr(space, (2,), 5.0)
        result = ones.outer(fives)
        assert result.components == (2, 2)
        expected = np.full((2, 2, *space.shape), 5.0)
        assert np.allclose(result.eval(), expected)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_outer_vector(self, space: Space):
        A = expr(space, (2, 3))
        v = expr(space, (2,))
        assert A.outer(v).components == (2, 3, 2)

    @pytest.mark.parametrize("space", SPACES)
    def test_vector_outer_matrix(self, space: Space):
        v = expr(space, (2,))
        A = expr(space, (2, 3))
        assert v.outer(A).components == (2, 2, 3)

    @pytest.mark.parametrize("space", SPACES)
    def test_matrix_outer_matrix(self, space: Space):
        A = expr(space, (2, 3))
        assert A.outer(A).components == (2, 3, 2, 3)
