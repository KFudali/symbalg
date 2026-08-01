import pytest
import numpy as np
from algebra.field import Field, to_operator
from algebra.space import Space, Shape
from tools.buffer import ConstValueBuffer

SPACES = [Space((10,)), Space((10, 10)), Space((10, 10, 10))]


def _comps(space: Space) -> list[tuple[int, ...]]:
    comps = [tuple(space.ndim for _ in range(i)) for i in range(3)]
    return comps


@pytest.mark.parametrize("space", SPACES)
def test_to_operator_shapes(space):
    for components in _comps(space):
        shape = Shape(space, components)
        field_values = np.random.uniform(0, 10, shape.fieldshape())
        field = Field(shape, ConstValueBuffer(field_values))
        operator = to_operator(field)

        assert operator.mat.shape == field.shape
        assert operator.mat.eval().shape == operator.mat.shape.sparseshape()


@pytest.mark.parametrize("space", SPACES)
def test_to_operator_apply(space):
    for components in _comps(space):
        shape = Shape(space, components)
        field_values = np.random.uniform(0, 10, shape.fieldshape())
        field = Field(shape, ConstValueBuffer(field_values))
        operator = to_operator(field)

        other_values = np.random.uniform(0, 10, shape.fieldshape())
        out = operator.apply_to(other_values)
        assert np.allclose(out, field_values * other_values)
