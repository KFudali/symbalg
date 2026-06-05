import pytest
import numpy as np
from discrete.fd.operators import dx
from algebra.space import Space


def test_grad_shapes():
    # Scalar field, 1D space -> vector field with 1 component
    shape = (10,)
    space = Space(shape)
    g = dx.grad(space, order=2, h=0.01)
    arr = np.ones(shape=shape)
    out = g.apply_to(arr)
    assert out.shape == (1, 10)

    # Scalar field, 2D space -> vector field with 2 components
    shape = (10, 10)
    space = Space(shape)
    g = dx.grad(space, order=2, h=0.01)
    arr = np.ones(shape=shape)
    out = g.apply_to(arr)
    assert out.shape == (2, 10, 10)

    # Tensor (rank 2) field, 2D space -> rank-3 tensor field
    arr = np.ones(shape=(3, 3, 10, 10))
    out = g.apply_to(arr)
    assert out.shape == (3, 3, 2, 10, 10)

    # Vector field with 1 outer component, 2D space
    arr = np.ones(shape=(1, 10, 10))
    out = g.apply_to(arr)
    assert out.shape == (1, 2, 10, 10)

    # Mismatched out shape should raise
    with pytest.raises(AssertionError):
        out = np.zeros(shape=(5, 5))
        arr = np.ones(shape=(1, 10, 10))
        g.apply(arr, out)


def test_grad_values():
    # Constant field -> gradient should be 0
    shape = (10, 10)
    space = Space(shape)
    g = dx.grad(space, order=2, h=0.01)
    arr = np.ones(shape=shape, dtype=float)
    out = g.apply_to(arr)
    assert np.allclose(out, 0.0, atol=1e-6)

    # Linear 1D field f(x) = x  ->  df/dx = 1
    x = np.arange(0.0, 10.01, 0.01)
    space = Space(x.shape)
    g = dx.grad(space, order=2, h=0.01)
    out = g.apply_to(x.copy())
    # out shape (1, N)
    assert np.allclose(out[0], 1.0, atol=1e-6)

    # Quadratic 1D field f(x) = x^2  ->  df/dx = 2x
    arr = (x * x).copy()
    out = g.apply_to(arr)
    assert np.allclose(out[0], 2.0 * x, atol=1e-6)
