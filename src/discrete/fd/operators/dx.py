from .core import FDLapLikeOperator, FDGradLikeOperator, FdDivLikeOperator
from discrete.fd.tools import dx, ddx, stencil
from algebra.space import Space


def laplace(space: Space, order: int, h: float) -> FDLapLikeOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = [ddx.stencil(order, h) for _ in range(space.ndim)]
    return FDLapLikeOperator(space, stencils)


def grad(space: Space, order: int, h: float) -> FDGradLikeOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return FDGradLikeOperator(space, stencils)


def div(space: Space, order: int, h: float) -> FDDivLikeOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = [dx.stencil(order, h) for _ in range(space.ndim)]
    return FDDivLikeOperator(space, stencils)


def eye(space: Space) -> FDLapLikeOperator:
    stencils = [stencil.AxStencil(stencil.Stencil({}), (), ())]
    stencils[0] = stencil.AxStencil(stencil.Stencil({0: 1.0}), (), ())
    return FDLapLikeOperator(space, stencils)
