import pytest
import numpy as np

import region
from region import Region


def test_full_region():
    arr_1d = np.array([1, 2, 3])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr_3d = np.arange(1, 28).reshape((3, 3, 3))
    assert np.array_equal(arr_1d, arr_1d[region.full(ndim=1)])
    assert np.array_equal(arr_2d, arr_2d[region.full(ndim=2)])
    assert np.array_equal(arr_3d, arr_3d[region.full(ndim=3)])


def test_interior_region():
    # one dim safe path behaviour
    assert Region.interior(1, (1,)) == Region((slice(1, -1),))
    assert Region.interior(1, ((2, 3),)) == Region((slice(2, -3),))

    # multi dim safe path behaviour
    assert Region.interior(
        2,
        (1, 2),
    ) == Region((slice(1, -1), slice(2, -2)))
    assert Region.interior(
        2,
        ((1, 2), (2, 1)),
    ) == Region((slice(1, -2), slice(2, -1)))

    # raises on negative offset
    with pytest.raises(ValueError):
        Region.interior(1, (-1,))
    with pytest.raises(ValueError):
        Region.interior(2, ((1, 2), (-1, 4)))

    # handles 0 offsets
    assert Region.interior(1, (0,)) == Region((slice(None, None),))
    assert Region.interior(1, ((0, 1),)) == Region((slice(None, -1),))
    assert Region.interior(1, ((2, 0),)) == Region((slice(2, None),))


def test_boundary_region():
    assert region.boundary(1, 0, -1) == Region((slice(None, 1),))
    assert region.boundary(1, 0, -1, True) == Region((slice(None, 1),))
    assert region.boundary(1, 0, 1) == Region((slice(-1, None),))
    assert region.boundary(1, 0, 1, True) == Region((slice(-1, None),))

    assert region.boundary(2, 0, -1) == Region((slice(None, 1), slice(None, None)))
    assert region.boundary(2, 0, -1, True) == Region((slice(None, 1), slice(1, -1)))