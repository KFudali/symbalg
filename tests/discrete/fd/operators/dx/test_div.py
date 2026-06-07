import pytest
import numpy as np
from discrete.fd.operators import dx
from algebra.space import Space


def test_div_shapes():
    shape = (10,)
    space = Space(shape)
    d = dx.div(space, order=2, h=0.01)

    # Vector field, 1D space (1 component) -> scalar field
    arr = np.ones(shape=(1, *shape))
    out = d.apply_to(arr)
    assert out.shape == shape

    # Vector field, 2D space (2 components) -> scalar field
    shape = (10, 10)
    space = Space(shape)
    d = dx.div(space, order=2, h=0.01)
    arr = np.ones(shape=(2, *shape))
    out = d.apply_to(arr)
    assert out.shape == shape

    # Tensor (outer rank 3) field, 2D space, contracting last rank axis
    # Shape (3, 2, 10, 10) -> (3, 10, 10)
    arr = np.ones(shape=(3, 2, 10, 10))
    out = d.apply_to(arr)
    assert out.shape == (3, 10, 10)

    # Higher-rank tensor: (3, 3, 2, 10, 10) -> (3, 3, 10, 10)
    arr = np.ones(shape=(3, 3, 2, 10, 10))
    out = d.apply_to(arr)
    assert out.shape == (3, 3, 10, 10)

    # Mismatched out shape should raise
    with pytest.raises(AssertionError):
        out = np.zeros(shape=(5, 5))
        arr = np.ones(shape=(2, 10, 10))
        d.apply(arr, out)

    # Wrong size of contracted axis should raise
    with pytest.raises(AssertionError):
        arr = np.ones(shape=(3, 10, 10))  # 3 != space_ndim=2
        d.apply_to(arr)


def test_div_values():
    shape = (10, 10)
    space = Space(shape)
    d = dx.div(space, order=2, h=0.01)

    # Constant vector field -> divergence is 0
    arr = np.ones(shape=(2, *shape), dtype=float)
    out = d.apply_to(arr)
    assert np.allclose(out, 0.0, atol=1e-6)

    # Linear 1D vector field v(x) = x  ->  div(v) = dv/dx = 1
    x = np.arange(0.0, 10.01, 0.01)
    space = Space(x.shape)
    d = dx.div(space, order=2, h=0.01)
    arr = x[np.newaxis, :].copy()  # shape (1, N)
    out = d.apply_to(arr)
    assert np.allclose(out, 1.0, atol=1e-6)

    # Quadratic 1D vector field v(x) = x^2 -> div(v) = 2x
    arr = (x * x)[np.newaxis, :].copy()
    out = d.apply_to(arr)
    assert np.allclose(out, 2.0 * x, atol=1e-6)
