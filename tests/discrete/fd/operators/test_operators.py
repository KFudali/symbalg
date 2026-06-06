import numpy as np
from discrete.fd.operators.dx import FDLapLikeOperator
from discrete.fd.tools.stencil import AxStencil, Stencil
from algebra.space import Space
from algebra.symbolic import SymbolicOperator


def laplike() -> FDLapLikeOperator:
    space = Space((10,))
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({0: -10.0})
    right = Stencil({0: 10.0})
    stencil = AxStencil(interior, (left,), (right,))
    return FDLapLikeOperator(space, (stencil,))


def interior_only() -> FDLapLikeOperator:
    space = Space((10,))
    interior = Stencil({0: 1.0})
    stencil = AxStencil(interior, (), ())
    return FDLapLikeOperator(space, (stencil,))


def test_wrapped_subtraction():
    first = SymbolicOperator[FDLapLikeOperator].wrap(laplike())
    second = SymbolicOperator[FDLapLikeOperator].wrap(laplike())

    sub = (first - second).resolve()
    ax_stencil = sub.stencils[0]
    stencils = (*ax_stencil.lefts, ax_stencil.interior, *ax_stencil.rights)

    for st in stencils:
        for w in st.weights.values():
            assert np.isclose(w, 0.0)


def test_wrapped_subtraction_diff_lefts_rights():
    first = SymbolicOperator[FDLapLikeOperator].wrap(laplike())
    second = SymbolicOperator[FDLapLikeOperator].wrap(interior_only())
    sub = (first - second).resolve()
    ax_stencil = sub.stencils[0]
    assert len(ax_stencil.lefts) == 1
    assert len(ax_stencil.rights) == 1
    assert ax_stencil.lefts[0].weights == {0: -11.0}
    assert ax_stencil.rights[0].weights == {0: 9.0}
    assert ax_stencil.interior.weights == {-1: 1.0, 0: 0.0, 1: 1.0}
