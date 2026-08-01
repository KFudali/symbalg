import pytest
import numpy as np

from algebra.space import utils
from algebra.space import Space, Shape


def test_pick_component_single_int():
    space = Space((10, 10))
    fieldshape = Shape(space, (3,))
    result = utils.pick_component(fieldshape, 1)
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == space.shape

    fieldshape = Shape(space, (3, 3))
    result = utils.pick_component(fieldshape, 1)
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == (3, *space.shape)

    with pytest.raises(ValueError):
        utils.pick_component(fieldshape, 4)


def test_pick_component_int_tuple():
    space = Space((10, 10))
    fieldshape = Shape(space, (3, 3))
    result = utils.pick_component(fieldshape, (1, 1))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == space.shape

    fieldshape = Shape(space, (3, 3, 3))
    result = utils.pick_component(fieldshape, (0, 1, 2))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == space.shape

    with pytest.raises(ValueError):
        utils.pick_component(fieldshape, (1, 2))


def test_pick_component_slice_tuple():
    space = Space((10, 10))
    fieldshape = Shape(space, (3,))
    result = utils.pick_component(fieldshape, (slice(0, 2),))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == (2, *space.shape)

    fieldshape = Shape(space, (3, 3))
    result = utils.pick_component(fieldshape, (slice(1, 3), slice(0, 2)))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == (2, 2, *space.shape)

    with pytest.raises(ValueError):
        utils.pick_component(fieldshape, (slice(0, 1), slice(0, 1), slice(0, 1)))


def test_pick_component_mixed_tuple():
    space = Space((10, 10))
    fieldshape = Shape(space, (3, 3))
    result = utils.pick_component(fieldshape, (1, slice(0, 2)))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == (2, *space.shape)

    fieldshape = Shape(space, (3, 3, 3))
    result = utils.pick_component(fieldshape, (slice(1, 3), 1, 2))
    arr = np.zeros(shape=fieldshape.fieldshape(), dtype=int)
    assert arr[result].shape == (2, *space.shape)
