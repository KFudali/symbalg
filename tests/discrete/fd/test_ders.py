import numpy as np

import discrete.fd.tools as ders


def test_dx():
    dx = ders.dx.stencil(1, 0.01)
    field = np.ones(shape=(100,), dtype=float)
    out = dx.eval(0, field)
    assert np.all(out < 1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = dx.eval(0, field)
    assert np.allclose(out, 1.0, atol=1e-6)

    dx = ders.dx.stencil(2, 0.01)
    field = np.arange(0.0, 1.01, 0.01)
    out = dx.eval(0, field * field)
    assert np.allclose(out, 2.0 * field, atol=1e-6)


def test_ddx():
    ddx = ders.ddx.stencil(2, 0.01)
    field = np.ones(shape=(100,), dtype=float)
    out = ddx.eval(0, field)
    assert np.all(out < 1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = ddx.eval(0, field)
    assert np.allclose(out, 0.0, atol=1e-6)

    field = np.arange(0.0, 1.01, 0.01)
    out = ddx.eval(0, field * field)
    assert np.allclose(out, 2.0, atol=1e-6)
