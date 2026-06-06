from discrete.fd.operators.dx import FDLapLikeOperator
from discrete.fd.tools.stencil import AxStencil, Stencil
from algebra.space import Space
from algebra.symbolic import SymbolicOperator


def laplike() -> FDLapLikeOperator:
    space = Space((10,))
    interior = Stencil({-1: 1.0, 0: 1.0, 1: 1.0})
    left = Stencil({-1: -10.0})
    right = Stencil({1: 10.0})
    stencil = AxStencil(interior, (left,), (right,))
    return FDLapLikeOperator(space, (stencil,))


def test_wrapped_magics():
    first = laplike()
    second = laplike()
