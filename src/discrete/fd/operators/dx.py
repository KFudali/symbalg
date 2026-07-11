from algebra.space import ShapeTransform
from discrete.fd.domain import FDDomain
from discrete.fd.tools import dx, ddx, stencil
from .fd_operator import FDOperator


def laplace(domain: FDDomain, order: int, h: float) -> FDOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(ddx.stencil(order, h) for _ in range(domain.space.ndim))
    return FDOperator(domain, ShapeTransform.NONE, stencils)


def grad(domain: FDDomain, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(domain.space.ndim))
    return FDOperator(domain, ShapeTransform.INCREASE_RANK, stencils)


def div(domain: FDDomain, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(domain.space.ndim))
    return FDOperator(domain, ShapeTransform.REDUCE_RANK, stencils)


def eye(domain: FDDomain) -> FDOperator:
    stencils = []
    for _ in range(domain.space.ndim):
        stencils.append(stencil.AxStencil(stencil.Stencil({}), (), ()))
    stencils[0] = stencil.AxStencil(stencil.Stencil({0: 1.0}), (), ())
    return FDOperator(domain, ShapeTransform.NONE, tuple(stencils))
