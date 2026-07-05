from algebra.exceptions import ShapeMismatchError
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
    Space((10,)),
    Space((10, 10)),
    Space((10, 10, 10)),
]


@pytest.mark.parametrize("space", SPACES)
def test_symbolic_expression_trace(space: Space):
    ones = expr(space, (space.ndim, space.ndim), 10.0)
    assert np.allclose(ones.trace().eval(), space.ndim * 10.0)

    ones = expr(space, (), 10.0)
    with pytest.raises(ShapeMismatchError):
        ones.trace()

    ones = expr(space, (space.ndim,), 10.0)
    with pytest.raises(ShapeMismatchError):
        ones.trace()

    ones = expr(space, (space.ndim, space.ndim, space.ndim), 10.0)
    with pytest.raises(ShapeMismatchError):
        ones.trace()
