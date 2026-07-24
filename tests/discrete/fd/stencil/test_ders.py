import numpy as np

from discrete.fd.stencil import dx, ddx


def test_dx():
    dx_stencil = dx.stencil(1, 0.01)
    field = np.ones(shape=(100,), dtype=float)
    out = dx_stencil.eval(0, field)
    assert np.all(out < 1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = dx_stencil.eval(0, field)
    assert np.allclose(out, 1.0, atol=1e-6)

    dx_stencil = dx.stencil(2, 0.01)
    field = np.arange(0.0, 1.01, 0.01)
    out = dx_stencil.eval(0, field * field)
    assert np.allclose(out, 2.0 * field, atol=1e-6)


def test_ddx():
    ddx_stencil = ddx.stencil(2, 0.01)
    field = np.ones(shape=(100,), dtype=float)
    out = ddx_stencil.eval(0, field)
    assert np.all(out < 1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = ddx_stencil.eval(0, field)
    assert np.allclose(out, 0.0, atol=1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = ddx_stencil.eval(0, field * field)
    assert np.allclose(out, 2.0, atol=1e-6)
