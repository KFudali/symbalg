import numpy as np
import pytest

from algebra.expression import ConstExpression
from algebra.symbolic import SymbolicExpression
from algebra.space import Space


def expr(
    space: Space,
    comps: tuple[int, ...],
    value: float = 0.0,
) -> SymbolicExpression:
    return SymbolicExpression.wrap(
        ConstExpression(
            space,
            np.ones((*comps, *space.shape)) * value,
        )
    )


SPACES = [
    Space((10, 10)),
    Space((10, 10, 10)),
]


@pytest.mark.parametrize("space", SPACES)
def test_expr_shape(space: Space):
    A = expr(space, (2, 3))

    assert A.comps == (2, 3)
    assert A.shape == (2, 3, *space.shape)


@pytest.mark.parametrize("space", SPACES)
def test_symbolic_expression_dot_scalar(space: Space):
    ones = expr(space, (), 1.0)
    fives = expr(space, (), 5.0)

    result = ones.dot(fives)

    assert result.comps == ()
    assert np.allclose(result.eval(), 5.0)


@pytest.mark.parametrize("space", SPACES)
def test_symbolic_expression_dot_vector(space: Space):
    ones = expr(space, (2,), 1.0)
    fives = expr(space, (2,), 5.0)

    result = ones.dot(fives)

    assert result.comps == ()
    assert np.allclose(result.eval(), 10.0)


@pytest.mark.parametrize("space", SPACES)
def test_symbolic_expression_inner_vector(space: Space):
    ones = expr(space, (2,), 1.0)
    fives = expr(space, (2,), 5.0)

    result = ones.inner(fives)

    assert result.comps == ()
    assert np.allclose(result.eval(), 10.0)


@pytest.mark.parametrize("space", SPACES)
def test_symbolic_expression_outer_vector(space: Space):
    ones = expr(space, (2,), 1.0)
    fives = expr(space, (2,), 5.0)

    result = ones.outer(fives)

    assert result.comps == (2, 2)

    expected = np.full((2, 2, *space.shape), 5.0)

    assert np.allclose(result.eval(), expected)


@pytest.mark.parametrize("space", SPACES)
def test_dot_shapes(space: Space):
    s = expr(space, ())
    v2 = expr(space, (2,))
    v3 = expr(space, (3,))
    A = expr(space, (2, 3))
    B = expr(space, (3, 4))
    C = expr(space, (4, 5))

    # scalar rules
    assert s.dot(s).comps == ()
    assert s.dot(v2).comps == (2,)
    assert v2.dot(s).comps == (2,)
    assert s.dot(A).comps == (2, 3)
    assert A.dot(s).comps == (2, 3)

    # vector
    assert v2.dot(v2).comps == ()

    # matrix-vector
    assert A.dot(v3).comps == (2,)
    assert v3.dot(B).comps == (4,)

    # matrix-matrix
    assert A.dot(B).comps == (2, 4)
    assert B.dot(C).comps == (3, 5)

    # chained multiplication
    assert A.dot(B).dot(C).comps == (2, 5)


@pytest.mark.parametrize("space", SPACES)
def test_outer_shapes(space: Space):
    s = expr(space, ())
    v = expr(space, (2,))
    A = expr(space, (2, 3))

    # scalar behaves like multiplication
    assert s.outer(s).comps == ()
    assert s.outer(v).comps == (2,)
    assert v.outer(s).comps == (2,)

    # vector outer product
    assert v.outer(v).comps == (2, 2)

    # tensor product
    assert A.outer(v).comps == (2, 3, 2)
    assert v.outer(A).comps == (2, 2, 3)
    assert A.outer(A).comps == (2, 3, 2, 3)


@pytest.mark.parametrize("space", SPACES)
def test_inner_shapes(space: Space):
    s = expr(space, ())
    v = expr(space, (2,))
    A = expr(space, (2, 3))

    # scalars
    assert s.inner(s).comps == ()

    # vectors
    assert v.inner(v).comps == ()

    # matrices (full contraction)
    assert A.inner(A).comps == ()

    # scalar/tensor
    assert s.inner(v).comps == (2,)
    assert v.inner(s).comps == (2,)
    assert s.inner(A).comps == (2, 3)
    assert A.inner(s).comps == (2, 3)