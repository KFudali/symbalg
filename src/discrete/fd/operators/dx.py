from algebra.space import Space, ShapeTransform
from discrete.fd.tools import dx, ddx, stencil
from .core import FDOperator


def laplace(space: Space, order: int, h: float) -> FDOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(ddx.stencil(order, h) for _ in range(space.ndim))
    return FDOperator(space, ShapeTransform.NONE, stencils)


def grad(space: Space, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return FDOperator(space, ShapeTransform.INCREASE_RANK, stencils)


def div(space: Space, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return FDOperator(space, ShapeTransform.REDUCE_RANK, stencils)


def eye(space: Space) -> FDOperator:
    stencils = []
    for _ in range(space.ndim):
        stencils.append(stencil.AxStencil(stencil.Stencil({}), (), ()))
    stencils[0] = stencil.AxStencil(stencil.Stencil({0: 1.0}), (), ())
    return FDOperator(space, ShapeTransform.NONE, tuple(stencils))
