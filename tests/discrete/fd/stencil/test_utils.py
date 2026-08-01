import pytest
import numpy as np

import discrete.fd.stencil as st
from algebra.space import Space

SPACES = [Space((10,))]


@pytest.mark.parametrize("space", SPACES)
def test_ax_stencil_to_array(space: Space):
    interior = st.Stencil({-1: 100.0, 0: 200.0, 1: 300.0})
    left = st.Stencil({0: 200.0, 1: 300.0})
    right = st.Stencil({-1: 100.0, 0: 200.0})
    ax_stencil = st.AxStencil(interior, (left,), (right,))

    sparse = st.utils.stencil_to_array(space, ax_stencil)

    field = np.ones(shape=space.shape, dtype=float)
    out = np.zeros_like(field)

    for ax in range(space.ndim):
        ax_stencil.eval_to(ax, field, out)

    assert np.allclose(out, sparse @ field.ravel())
