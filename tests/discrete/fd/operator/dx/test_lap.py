import pytest
import numpy as np
from discrete.fd.operators import dx
from algebra.space import Space


def test_lap_shapes():
    shape = (10,)
    space = Space(shape)
    lap = dx.laplace(space, order=2, h=0.01)
    arr = np.ones(shape=shape)
    out = lap.apply_to(arr)
    assert out.shape == arr.shape

    shape = (10, 10)
    space = Space(shape)
    lap = dx.laplace(space, order=2, h=0.01)
    arr = np.ones(shape=shape)
    out = lap.apply_to(arr)
    assert out.shape == arr.shape

    arr = np.ones(shape=(3, 3, 10, 10))
    out = lap.apply_to(arr)
    assert out.shape == arr.shape

    arr = np.ones(shape=(1, 10, 10))
    out = lap.apply_to(arr)
    assert out.shape == arr.shape

    with pytest.raises(AssertionError):
        out = np.zeros(shape=(5, 5))
        arr = np.ones(shape=(1, 10, 10))
        lap.apply(arr, out)


def test_lap_values():
    # Constant field
    shape = (10, 10)
    space = Space(shape)
    lap = dx.laplace(space, order=2, h=0.01)
    arr = np.ones(shape=shape, dtype=float)
    out = lap.apply_to(arr)
    assert np.allclose(out, 0.0, atol=1e-6)

    # Linear field
    x = np.arange(0.0, 10.01, 0.01)
    space = Space(x.shape)
    lap = dx.laplace(space, order=2, h=0.01)
    out = lap.apply_to(x.copy())
    assert np.allclose(out, 0.0, atol=1e-6)

    # Square field
    arr = (x * x).copy()
    out = lap.apply_to(arr)
    assert np.allclose(out, 2.0, atol=1e-6)


def test_lap_magics():
    shape = (10, 10)
    space = Space(shape)
    l_lap = dx.laplace(space, order=2, h=0.01)
    r_lap = dx.laplace(space, order=2, h=0.02)

    binary_ops = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]
    for binary_op in binary_ops:
        combined = binary_op(l_lap, r_lap)
        assert isinstance(combined, type(l_lap))
        assert combined.space.ndim == l_lap.space.ndim
        for ax in range(l_lap.space.ndim):
            expected = binary_op(l_lap.stencils[ax], r_lap.stencils[ax])
            assert combined.stencils[ax].interior == expected.interior
            assert combined.stencils[ax].lefts == expected.lefts
            assert combined.stencils[ax].rights == expected.rights

    # Scalar scaling
    scaled = l_lap * 2.0
    assert isinstance(scaled, type(l_lap))
    for ax in range(l_lap.space.ndim):
        assert scaled.stencils[ax].interior == l_lap.stencils[ax].interior * 2.0

    scaled = 2.0 * l_lap
    for ax in range(l_lap.space.ndim):
        assert scaled.stencils[ax].interior == l_lap.stencils[ax].interior * 2.0

    scaled = l_lap / 2.0
    for ax in range(l_lap.space.ndim):
        assert scaled.stencils[ax].interior == l_lap.stencils[ax].interior * 0.5
