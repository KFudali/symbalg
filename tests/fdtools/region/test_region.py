import pytest
import numpy as np

from region import Region, ShiftOutsideBounds


def test_region_access():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    reg = Region((slice(0, 1), slice(None, None)))
    assert np.array_equal(arr[reg], np.array([[1, 2, 3]]))


def test_shift():
    reg = Region((slice(2, 3), slice(3, 4)))
    # standard behaviour
    assert reg.shift(0, 1) == Region((slice(3, 4), slice(3, 4)))
    assert reg.shift(0, -1) == Region((slice(1, 2), slice(3, 4)))

    assert reg.shift(1, 1) == Region((slice(2, 3), slice(4, 5)))
    assert reg.shift(1, -1) == Region((slice(2, 3), slice(2, 3)))

    # works with negatives
    reg = Region((slice(2, -1), slice(-5, None)))
    assert reg.shift(0, 1) == Region((slice(3, None), slice(-5, None)))
    assert reg.shift(0, -1) == Region((slice(1, -2), slice(-5, None)))
    assert reg.shift(1, -1) == Region((slice(2, -1), slice(-6, -1)))

    # raises on shift outside bounds
    reg = Region((slice(2, -1),))
    with pytest.raises(ShiftOutsideBounds):
        reg.shift(0, -3)
    with pytest.raises(ShiftOutsideBounds):
        assert reg.shift(0, 2)
    with pytest.raises(ShiftOutsideBounds):
        Region((slice(None, 2),)).shift(0, -1)
    with pytest.raises(ShiftOutsideBounds):
        Region((slice(2, None),)).shift(0, 1)

    # no effect for full region
    reg = Region((slice(None, None),))
    assert reg.shift(0, 1) == reg
    assert reg.shift(0, -1) == reg

    # If start or stop is None it shifts to a number
    assert Region((slice(None, 2),)).shift(0, 1) == Region((slice(1, 3),))
    assert Region((slice(2, None),)).shift(0, -1) == Region((slice(1, -1),))


def test_extend():
    reg = Region((slice(2, 3), slice(3, 4)))
    # standard behaviour
    assert reg.extend(0, 1) == Region((slice(2, 4), slice(3, 4)))
    assert reg.extend(0, -1) == Region((slice(1, 3), slice(3, 4)))

    assert reg.extend(1, 2) == Region((slice(2, 3), slice(3, 6)))
    assert reg.extend(1, -2) == Region((slice(2, 3), slice(1, 4)))

    # works with negatives
    reg = Region((slice(2, -1), slice(-5, None)))
    assert reg.extend(0, 1) == Region((slice(2, None), slice(-5, None)))
    assert reg.extend(1, -3) == Region((slice(2, -1), slice(-8, None)))

    # extend clamps to None
    reg = Region((slice(2, -2),))
    assert reg.extend(0, -3) == Region((slice(None, -2),))
    assert reg.extend(0, 3) == Region((slice(2, None),))

    # no effect for full region
    reg = Region((slice(None, None),))
    assert reg.extend(0, 1) == reg
    assert reg.extend(0, -1) == reg

    # works with empty region
    reg = Region((slice(0, 0),))
    assert reg.extend(0, 5) == Region((slice(0, 5),))
    assert reg.extend(0, -3) == Region((slice(0, 0),))


def test_trim():
    reg = Region((slice(2, 5), slice(3, 6)))
    # standard behaviour
    assert reg.trim(0, 1) == Region((slice(2, 4), slice(3, 6)))
    assert reg.trim(0, -1) == Region((slice(3, 5), slice(3, 6)))

    assert reg.trim(1, 2) == Region((slice(2, 5), slice(3, 4)))
    assert reg.trim(1, -2) == Region((slice(2, 5), slice(5, 6)))

    # works with negatives
    reg = Region((slice(2, -1), slice(-6, None)))
    assert reg.trim(0, 1) == Region((slice(2, -2), slice(-6, None)))
    assert reg.trim(1, -3) == Region((slice(2, -1), slice(-3, None)))

    # trim clamps to start to stop
    reg = Region((slice(2, 4),))
    assert reg.trim(0, -3) == Region((slice(4, 4),))
    assert reg.trim(0, 3) == Region((slice(2, 2),))

    # trims full region
    reg = Region((slice(None, None),))
    assert reg.trim(0, 3) == Region((slice(None, -3),))
    assert reg.trim(0, -2) == Region((slice(2, None),))

    # no effect with empty region
    reg = Region((slice(0, 0),))
    assert reg.trim(0, 5) == Region((slice(0, 0),))
    assert reg.trim(0, -3) == Region((slice(0, 0),))


def test_normalize():
    reg = Region((slice(None, 8), slice(4, 12), slice(None, None)))
    normalized = reg.normalize((4, 6, 8))

    assert normalized[0] == slice(0, 4)
    assert normalized[1] == slice(4, 6)
    assert normalized[2] == slice(0, 8)


def test_intersect():
    reg_a = Region((slice(None, None), slice(1, 5)))
    reg_b = Region((slice(2, 3), slice(3, 4)))
    assert reg_a.intersect(reg_b) == reg_b.intersect(reg_a) == reg_b

    reg_a = Region((slice(2, 5), slice(1, 3)))
    reg_b = Region((slice(4, 10), slice(2, 6)))
    expected = Region((slice(4, 5), slice(2, 3)))
    assert reg_a.intersect(reg_b) == reg_a.intersect(reg_b) == expected

    reg_a = Region((slice(2, 4), slice(4, 6)))
    reg_b = Region((slice(None, None), slice(1, 2)))
    expected = Region((slice(2, 4), slice(0, 0)))
    assert reg_a.intersect(reg_b) == reg_a.intersect(reg_b) == expected
