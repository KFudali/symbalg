# import numpy as np
# from discrete.fd.operator import dx, as_array
# from algebra.space import Space
#
#
# def test_laplace_to_array():
#     space = Space((10, 10))
#     laplace = dx.laplace(space, 2, 0.01)
#     array_form = as_array.as_array(laplace)
#     x = np.random.uniform(0, 10, space.shape)
#     np.testing.assert_allclose(laplace.apply_to(x), array_form.apply_to(x))
