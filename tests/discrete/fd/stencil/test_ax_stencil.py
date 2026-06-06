import pytest
import numpy as np
import discrete.fd.tools.stencil as st


def test_ax_stencil_init():
    interior = st.Stencil({-1: 100.0, 0: 200.0, 1: 300.0})
    left = st.Stencil({0: 200.0, 1: 300.0})
    right = st.Stencil({-1: 100.0, 0: 200.0})

    st.AxStencil(interior, (left,), (right,))
    bad_left = interior.copy()
    bad_right = interior.copy()

    with pytest.raises(AssertionError):
        st.AxStencil(interior, (bad_left,), (right,))

    with pytest.raises(AssertionError):
        st.AxStencil(interior, (left,), (bad_right,))


def test_ax_stencil_eval_1d():
    interior = st.Stencil({-1: 100.0, 0: -200.0, 1: 100.0})
    left = st.Stencil({0: 500.0, 1: 100.0})
    right = st.Stencil({-1: 100.0, 0: 500.0})
    dx = st.AxStencil(interior, (left,), (right,))

    ones = np.ones((4,), dtype=float)
    out = np.zeros_like(ones, dtype=float)
    dx.eval_to(0, ones, out)
    assert np.isclose(out[0], 600.0)
    assert np.isclose(out[1], 0.0)
    assert np.isclose(out[2], 0.0)
    assert np.isclose(out[3], 600.0)


def test_ax_stencil_eval_nd():
    interior = st.Stencil({-1: 100.0, 0: -200.0, 1: 100.0})
    left = st.Stencil({0: 500.0, 1: 100.0})
    right = st.Stencil({-1: 100.0, 0: 500.0})
    dx = st.AxStencil(interior, (left,), (right,))

    ones = np.ones((4, 2), dtype=float)
    ones[:, 1] = 0
    ax_0_out = np.zeros_like(ones, dtype=float)
    ax_1_out = np.zeros_like(ones, dtype=float)
    dx.eval_to(0, ones, ax_0_out)
    dx.eval_to(1, ones, ax_1_out)
    assert np.allclose(
        ax_0_out, np.array([[600.0, 0.0], [0.0, 0.0], [0.0, 0.0], [600.0, 0.0]])
    )
    assert np.allclose(
        ax_1_out,
        np.array([[500.0, 100.0], [500.0, 100.0], [500.0, 100.0], [500.0, 100.0]]),
    )


def test_ax_stencil_magics():
    l_int = st.Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    l_left = st.Stencil({0: 1.0, 1: 1.0})
    l_right = st.Stencil({-1: 1.0, 0: 1.0})
    l_dx = st.AxStencil(l_int, (l_left,), (l_right,))

    r_int = st.Stencil({-1: 2.0, 0: 2.0, 1: 2.0})
    r_left = st.Stencil({0: 2.0, 1: 2.0})
    r_right = st.Stencil({-1: 2.0, 0: 2.0})
    r_dx = st.AxStencil(r_int, (r_left,), (r_right,))

    binary_ops = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ]
    for binary_op in binary_ops:
        assert binary_op(l_dx, r_dx).interior == binary_op(l_int, r_int)
        assert binary_op(l_dx, r_dx).lefts == (binary_op(l_left, r_left),)
        assert binary_op(l_dx, r_dx).rights == (binary_op(l_right, r_right),)

def test_norm_offsets():
    left_norm = st.Stencil({0, 10.0})
    right_norm = st.Stencil({0, -10.0})
    lefts = (st.Stencil({0: 1.0}), st.Stencil({0: 1.0}), st.Stencil({0: 1.0}))
    rights = (st.Stencil({0: 5.0}),)
    new_lefts, new_rights = st.AxStencil.norm_offset(lefts, left_norm, rights, right_norm)

    assert len(new_lefts) == len(new_rights)
    assert new_rights == (*rights, right_norm.copy(), right_norm.copy())

    lefts = (st.Stencil({0: 5.0}),)
    rights = (st.Stencil({0: 1.0}), st.Stencil({0: 1.0}), st.Stencil({0: 1.0}))
    new_lefts, new_rights = st.AxStencil.norm_offset(lefts, left_norm, rights, right_norm)
    
    assert len(new_lefts) == len(new_rights)
    assert new_lefts == (*lefts, left_norm.copy(), left_norm.copy())

def test_ax_stencil_mismatch_magics():
    l_int = st.Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    l_left = st.Stencil({0: 1.0, 1: 1.0})
    l_right = st.Stencil({-1: 1.0, 0: 1.0})
    l_dx = st.AxStencil(l_int, (l_left,), (l_right,))

    r_int = st.Stencil({-1: 2.0, 0: 2.0, 1: 2.0})
    r_left = st.Stencil({0: 2.0, 1: 2.0})
    r_right = st.Stencil({-1: 2.0, 0: 2.0})
    r_dx = st.AxStencil(r_int, (r_left,), (r_right,))