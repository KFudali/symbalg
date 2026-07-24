from algebra.space import Space, ShapeTransform

from discrete.fd.stencil import dx, ddx, AxStencil, Stencil, StencilOperator


def laplace(space: Space, order: int, h: float) -> StencilOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(ddx.stencil(order, h) for _ in range(space.ndim))
    return StencilOperator(space, ShapeTransform.NONE, stencils)


def grad(space: Space, order: int, h: float) -> StencilOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return StencilOperator(space, ShapeTransform.INCREASE_RANK, stencils)


def div(space: Space, order: int, h: float) -> StencilOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(space.ndim))
    return StencilOperator(space, ShapeTransform.REDUCE_RANK, stencils)


def eye(space: Space) -> StencilOperator:
    stencils = []
    for _ in range(space.ndim):
        stencils.append(AxStencil(Stencil({}), (), ()))
    stencils[0] = AxStencil(Stencil({0: 1.0}), (), ())
    return StencilOperator(space, ShapeTransform.NONE, tuple(stencils))
