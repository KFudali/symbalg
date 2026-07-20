from algebra.space import Space, ShapeTransform
from .stencil import dx, ddx, AxStencil, Stencil
from .fd_stencil_operator import FDStencilOperator


def laplace(space: Space, order: int, h: float) -> FDStencilOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(ddx.stencil(order, h) for _ in range(space.ndim))
    return FDStencilOperator(space, ShapeTransform.NONE, stencils)


def grad(space: Space, order: int, h: float) -> FDStencilOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return FDStencilOperator(space, ShapeTransform.INCREASE_RANK, stencils)


def div(space: Space, order: int, h: float) -> FDStencilOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return FDStencilOperator(space, ShapeTransform.REDUCE_RANK, stencils)


def eye(space: Space) -> FDStencilOperator:
    stencils = []
    for _ in range(space.ndim):
        stencils.append(AxStencil(Stencil({}), (), ()))
    stencils[0] = AxStencil(Stencil({0: 1.0}), (), ())
    return FDStencilOperator(space, ShapeTransform.NONE, tuple(stencils))
