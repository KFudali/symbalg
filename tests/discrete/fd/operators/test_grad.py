import pytest
import numpy as np
# import operators


# def test_grad_shapes():
#     g = operators.grad(order=2, h=0.01, ndim=1)

#     # Scalar field, 1D space -> vector field with 1 component
#     arr = np.ones(shape=(10,))
#     out = g.apply(arr)
#     assert out.shape == (1, 10)

#     # Scalar field, 2D space -> vector field with 2 components
#     g = operators.grad(order=2, h=0.01, ndim=2)
#     arr = np.ones(shape=(10, 10))
#     out = g.apply(arr)
#     assert out.shape == (2, 10, 10)

#     # Tensor (rank 2) field, 2D space -> rank-3 tensor field
#     arr = np.ones(shape=(3, 3, 10, 10))
#     out = g.apply(arr)
#     assert out.shape == (3, 3, 2, 10, 10)

#     # Vector field with 1 outer component, 2D space
#     arr = np.ones(shape=(1, 10, 10))
#     out = g.apply(field=arr)
#     assert out.shape == (1, 2, 10, 10)

#     # Mismatched out shape should raise
#     with pytest.raises(AssertionError):
#         out = np.zeros(shape=(5, 5))
#         arr = np.ones(shape=(1, 10, 10))
#         g.apply_to(arr, out)


# def test_grad_values():
#     g = operators.grad(order=2, h=0.01, ndim=2)

#     # Constant field -> gradient should be 0
#     arr = np.ones(shape=(10, 10), dtype=float)
#     out = g.apply(arr)
#     assert np.allclose(out, 0.0, atol=1e-6)

#     # Linear 1D field f(x) = x  ->  df/dx = 1
#     g = operators.grad(order=2, h=0.01, ndim=1)
#     arr = np.arange(0.0, 10.01, 0.01)
#     out = g.apply(arr)
#     # out shape (1, N)
#     assert np.allclose(out[0], 1.0, atol=1e-6)

#     # Quadratic 1D field f(x) = x^2  ->  df/dx = 2x
#     x = np.arange(0.0, 10.01, 0.01)
#     arr = x * x
#     out = g.apply(arr)
#     assert np.allclose(out[0], 2.0 * x, atol=1e-6)
