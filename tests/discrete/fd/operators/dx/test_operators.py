import numpy as np
from discrete.fd.operators.fd_operator import FDOperator
from discrete.fd.domain import FDDomain
from discrete.fd.tools.stencil import AxStencil, Stencil
from algebra.space import ShapeTransform
from algebra.symbolic import SymbolicOperator
from tools.geometry import StructuredGridND


def _domain(shape: tuple[int, ...]) -> FDDomain:
    return FDDomain(StructuredGridND(shape, (1.0,) * len(shape)))


def laplike() -> FDOperator:
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({0: -10.0})
    right = Stencil({0: 10.0})
    stencil = AxStencil(interior, (left,), (right,))
    return FDOperator(_domain((10,)), ShapeTransform.NONE, (stencil,))


def interior_only() -> FDOperator:
    interior = Stencil({0: 1.0})
    stencil = AxStencil(interior, (), ())
    return FDOperator(_domain((10,)), ShapeTransform.NONE, (stencil,))


def test_wrapped_subtraction():
    first = SymbolicOperator[FDOperator].wrap(laplike())
    second = SymbolicOperator[FDOperator].wrap(laplike())

    sub = (first - second).resolve()
    ax_stencil = sub.stencils[0]
    stencils = (*ax_stencil.lefts, ax_stencil.interior, *ax_stencil.rights)

    for st in stencils:
        for w in st.weights.values():
            assert np.isclose(w, 0.0)


def test_wrapped_subtraction_diff_lefts_rights():
    first = SymbolicOperator[FDOperator].wrap(laplike())
    second = SymbolicOperator[FDOperator].wrap(interior_only())
    sub = (first - second).resolve()
    ax_stencil = sub.stencils[0]
    assert len(ax_stencil.lefts) == 1
    assert len(ax_stencil.rights) == 1
    assert ax_stencil.lefts[0].weights == {0: -11.0}
    assert ax_stencil.rights[0].weights == {0: 9.0}
    assert ax_stencil.interior.weights == {-1: 1.0, 0: 0.0, 1: 1.0}
