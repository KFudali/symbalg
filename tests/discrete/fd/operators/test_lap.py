import pytest
import numpy as np
# import operators


# def test_lap_shapes():
#     lap = operators.laplace(order=2, h=0.01, ndim=1)
#     arr = np.ones(shape=(10,))
#     out = lap.apply(arr)
#     assert out.shape == arr.shape

#     lap = operators.laplace(order=2, h=0.01, ndim=2)
#     arr = np.ones(shape=(10, 10))
#     out = lap.apply(arr)
#     assert out.shape == arr.shape

#     arr = np.ones(shape=(3, 3, 10, 10))
#     out = lap.apply(arr)
#     assert out.shape == arr.shape

#     arr = np.ones(shape=(1, 10, 10))
#     out = lap.apply(arr)
#     assert out.shape == arr.shape

#     with pytest.raises(AssertionError):
#         out = np.zeros(shape=(5, 5))
#         arr = np.ones(shape=(1, 10, 10))
#         lap.apply_to(arr, out)


# def test_lap_values():
#     # Constant field
#     lap = operators.laplace(order=2, h=0.01, ndim=2)
#     arr = np.ones(shape=(10, 10), dtype=float)
#     out = lap.apply(arr)
#     assert np.allclose(out, 0.0, atol=1e-6)

#     # Linear field
#     lap = operators.laplace(order=2, h=0.01, ndim=1)
#     arr = np.arange(0.0, 10.01, 0.01)
#     out = lap.apply(arr)
#     assert np.allclose(out, 0.0, atol=1e-6)

#     # Square field
#     lap = operators.laplace(order=2, h=0.01, ndim=1)
#     arr = np.arange(0.0, 10.01, 0.01)
#     arr = arr * arr
#     out = lap.apply(arr)
#     assert np.allclose(out, 2.0, atol=1e-6)


# def test_lap_magics():
#     l_lap = operators.laplace(order=2, h=0.01, ndim=2)
#     r_lap = operators.laplace(order=2, h=0.02, ndim=2)

#     binary_ops = [
#         lambda a, b: a + b,
#         lambda a, b: a - b,
#         lambda a, b: a * b,
#         lambda a, b: a / b,
#     ]
#     for binary_op in binary_ops:
#         combined = binary_op(l_lap, r_lap)
#         assert isinstance(combined, type(l_lap))
#         assert combined.ndim == l_lap.ndim
#         for ax in range(l_lap.ndim):
#             expected = binary_op(l_lap.ax_stencils[ax], r_lap.ax_stencils[ax])
#             assert combined.ax_stencils[ax].interior == expected.interior
#             assert combined.ax_stencils[ax].lefts == expected.lefts
#             assert combined.ax_stencils[ax].rights == expected.rights

#     # Scalar scaling
#     scaled = l_lap * 2.0
#     assert isinstance(scaled, type(l_lap))
#     for ax in range(l_lap.ndim):
#         assert scaled.ax_stencils[ax].interior == l_lap.ax_stencils[ax].interior * 2.0

#     scaled = 2.0 * l_lap
#     for ax in range(l_lap.ndim):
#         assert scaled.ax_stencils[ax].interior == l_lap.ax_stencils[ax].interior * 2.0

#     scaled = l_lap / 2.0
#     for ax in range(l_lap.ndim):
#         assert scaled.ax_stencils[ax].interior == l_lap.ax_stencils[ax].interior * 0.5
