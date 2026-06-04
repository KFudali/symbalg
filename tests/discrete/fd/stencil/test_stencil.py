import numpy as np
import discrete.fd.tools.stencil as st


def test_stencil():
    dx = st.Stencil({-2: 100.0, 0: 200.0, 1: 300.0})
    assert dx.max_offsets() == (2, 1)
    assert dx.left_offsets() == [2]
    assert dx.right_offsets() == [1]

    ones = np.ones((4,), dtype=float)
    out = np.array([0.0])

    dx.eval_to(0, ones, out)
    assert np.isclose(out[0], 600.0)
    assert np.array_equal(out, dx.eval(0, ones))


def test_stencil_combine_add():
    left = st.Stencil({0: 1.0, 1: 5.0})
    right = st.Stencil({-1: 2.0, 0: 1.0, 1: -3.0})
    assert np.isclose((left + right).weights[-1], 2.0)
    assert np.isclose((left + right).weights[0], 2.0)
    assert np.isclose((left + right).weights[1], 2.0)


def test_stencil_combine_sub():
    left = st.Stencil({0: 1.0, 1: 5.0})
    right = st.Stencil({-1: 2.0, 0: 1.0, 1: -3.0})
    assert np.isclose((left - right).weights[-1], 2.0)
    assert np.isclose((left - right).weights[0], 0.0)
    assert np.isclose((left - right).weights[1], 8.0)


def test_stencil_combine_mul():
    left = st.Stencil({0: 1.0, 1: 5.0})
    right = st.Stencil({-1: 2.0, 0: 1.0, 1: -3.0})
    assert np.isclose((left * right).weights[-1], 0.0)
    assert np.isclose((left * right).weights[0], 1.0)
    assert np.isclose((left * right).weights[1], -15.0)


def test_stencil_combine_div():
    left = st.Stencil({-1: 1.0, 0: 1.0, 1: 5.0})
    right = st.Stencil({-1: 5.0, 0: 1.0, 1: -2.5})
    assert np.isclose((left / right).weights[-1], 0.2)
    assert np.isclose((left / right).weights[0], 1.0)
    assert np.isclose((left / right).weights[1], -2.0)
