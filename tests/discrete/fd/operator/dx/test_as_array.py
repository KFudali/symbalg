import numpy as np
import scipy.sparse as sp
from discrete.fd.operators.core import FDOperator
from discrete.fd.tools.stencil import AxStencil, Stencil
from algebra.space import Space, ShapeTransform


def _verify(op: FDOperator, x: np.ndarray):
    """Assert that as_array() @ x.ravel() == apply_to(x).ravel()."""
    M = op.as_array()
    assert isinstance(M, sp.spmatrix)
    expected = op.apply_to(x).ravel()
    actual = M @ x.ravel()
    assert np.allclose(actual, expected), (
        f"Mismatch for shape_transform={op.shape_transform}, "
        f"space={op.space.shape}\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )


def _laplike(shape: tuple[int, ...]) -> FDOperator:
    space = Space(shape)
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({0: -10.0})
    right = Stencil({0: 10.0})
    stencils = tuple(AxStencil(interior, (left,), (right,)) for _ in range(len(shape)))
    return FDOperator(space, ShapeTransform.NONE, stencils)


# ── NONE ─────────────────────────────────────────────────────────────


def test_none_1d():
    op = _laplike((10,))
    x = np.arange(10.0)
    _verify(op, x)


def test_none_2d():
    op = _laplike((8, 8))
    x = np.random.randn(8, 8)
    _verify(op, x)


def test_none_3d():
    op = _laplike((6, 6, 6))
    x = np.random.randn(6, 6, 6)
    _verify(op, x)


# ── INCREASE_RANK ────────────────────────────────────────────────────


def _gradlike(shape: tuple[int, ...]) -> FDOperator:
    space = Space(shape)
    interior = Stencil({-1: -0.5, 1: 0.5})
    left_fwd = Stencil({0: -1.0, 1: 1.0})
    right_bwd = Stencil({0: 1.0, -1: -1.0})
    stencils = tuple(
        AxStencil(interior, (left_fwd,), (right_bwd,)) for _ in range(len(shape))
    )
    return FDOperator(space, ShapeTransform.INCREASE_RANK, stencils)


def test_increase_rank_1d():
    op = _gradlike((10,))
    x = np.arange(10.0)
    _verify(op, x)


def test_increase_rank_2d():
    op = _gradlike((8, 8))
    x = np.random.randn(8, 8)
    _verify(op, x)


def test_increase_rank_3d():
    op = _gradlike((6, 6, 6))
    x = np.random.randn(6, 6, 6)
    _verify(op, x)


# ── REDUCE_RANK ──────────────────────────────────────────────────────


def _divlike(shape: tuple[int, ...]) -> FDOperator:
    space = Space(shape)
    interior = Stencil({-1: -0.5, 1: 0.5})
    left_fwd = Stencil({0: -1.0, 1: 1.0})
    right_bwd = Stencil({0: 1.0, -1: -1.0})
    stencils = tuple(
        AxStencil(interior, (left_fwd,), (right_bwd,)) for _ in range(len(shape))
    )
    return FDOperator(space, ShapeTransform.REDUCE_RANK, stencils)


def test_reduce_rank_1d():
    op = _divlike((10,))
    x = np.random.randn(1, 10)
    _verify(op, x)


def test_reduce_rank_2d():
    op = _divlike((8, 8))
    x = np.random.randn(2, 8, 8)
    _verify(op, x)


# ── BOUNDARY STENCILS ────────────────────────────────────────────────


def test_asymmetric_boundaries():
    """Multiple left/right boundary stencils with different stencil widths."""
    space = Space((10,))
    interior = Stencil({-2: 0.5, -1: -2.0, 0: 3.0, 1: -2.0, 2: 0.5})
    left0 = Stencil({0: 2.0, 1: -5.0, 2: 4.0, 3: -1.0})
    left1 = Stencil({-1: -0.5, 0: 1.0, 1: 0.5})
    right0 = Stencil({0: 2.0, -1: -5.0, -2: 4.0, -3: -1.0})
    right1 = Stencil({0: 1.0, -1: -0.5, 1: 0.5})
    ax_stencil = AxStencil(interior, (left0, left1), (right0, right1))
    op = FDOperator(space, ShapeTransform.NONE, (ax_stencil,))
    x = np.random.randn(10)
    _verify(op, x)


# ── COMBINED / SCALED ────────────────────────────────────────────────


def test_combined_operators():
    space = Space((10,))
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({0: -10.0})
    right = Stencil({0: 10.0})
    ax_stencil = AxStencil(interior, (left,), (right,))
    op1 = FDOperator(space, ShapeTransform.NONE, (ax_stencil,))
    op2 = FDOperator(space, ShapeTransform.NONE, (ax_stencil,))
    combined = op1 + 2.0 * op2
    x = np.random.randn(10)
    _verify(combined, x)


def test_scaled_operator():
    space = Space((10,))
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({0: -10.0})
    right = Stencil({0: 10.0})
    ax_stencil = AxStencil(interior, (left,), (right,))
    op = FDOperator(space, ShapeTransform.NONE, (ax_stencil,))
    scaled = 3.0 * op
    x = np.random.randn(10)
    _verify(scaled, x)


# ── SPARSITY ─────────────────────────────────────────────────────────


def test_sparsity_pattern():
    """Verify the matrix has expected non-zero structure for a 1D laplace."""
    op = _laplike((10,))
    M = op.as_array()
    # Row 0: {0} (1)
    # Rows 1-8: {i-1, i, i+1} (3 each = 24)
    # Row 9: {9} (1)
    # Total: 1 + 24 + 1 = 26
    assert M.nnz == 26, f"Expected 26 non-zeros, got {M.nnz}"
