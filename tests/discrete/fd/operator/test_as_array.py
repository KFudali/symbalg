import numpy as np
import pytest
from discrete.fd.operator import dx, as_array
from algebra.space import Space

spaces = [Space((10,)), Space((10, 10)), Space((10, 10, 10))]


@pytest.mark.parametrize("space", spaces)
def test_laplace_to_array(space):
    laplace = dx.laplace(space, 2, 0.01)
    array_form = as_array.as_array(laplace)
    x = np.random.uniform(0, 10, space.shape)
    np.testing.assert_allclose(laplace.apply_to(x), array_form.apply_to(x))


@pytest.mark.parametrize("space", spaces)
def test_grad_to_array(space):
    grad = dx.grad(space, 2, 0.01)
    array_form = as_array.as_array(grad)
    x = np.random.uniform(0, 10, space.shape)
    np.testing.assert_allclose(grad.apply_to(x), array_form.apply_to(x))


@pytest.mark.parametrize("space", spaces)
def test_div_to_array(space):
    div = dx.div(space, 2, 0.01)
    array_form = as_array.as_array(div)
    x = np.random.uniform(0, 10, (space.ndim, *space.shape))
    np.testing.assert_allclose(div.apply_to(x), array_form.apply_to(x))
