from algebra.space import Space, ShapeTransform
from discrete.fd.domain import FDDomain
from discrete.fd.tools import dx, ddx, stencil
from .fd_operator import FDOperator

SpaceOrDomain = Space | FDDomain


def _space(space_or_domain: SpaceOrDomain) -> Space:
    if isinstance(space_or_domain, FDDomain):
        return space_or_domain.space
    return space_or_domain


def laplace(space_or_domain: SpaceOrDomain, order: int, h: float) -> FDOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    s = _space(space_or_domain)
    stencils = tuple(ddx.stencil(order, h) for _ in range(s.ndim))
    return FDOperator(space_or_domain, ShapeTransform.NONE, stencils)


def grad(space_or_domain: SpaceOrDomain, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    s = _space(space_or_domain)
    stencils = tuple(dx.stencil(order, h) for _ in range(s.ndim))
    return FDOperator(space_or_domain, ShapeTransform.INCREASE_RANK, stencils)


def div(space_or_domain: SpaceOrDomain, order: int, h: float) -> FDOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    s = _space(space_or_domain)
    stencils = tuple(dx.stencil(order, h) for _ in range(s.ndim))
    return FDOperator(space_or_domain, ShapeTransform.REDUCE_RANK, stencils)


def eye(space_or_domain: SpaceOrDomain) -> FDOperator:
    s = _space(space_or_domain)
    stencils = []
    for _ in range(s.ndim):
        stencils.append(stencil.AxStencil(stencil.Stencil({}), (), ()))
    stencils[0] = stencil.AxStencil(stencil.Stencil({0: 1.0}), (), ())
    return FDOperator(space_or_domain, ShapeTransform.NONE, tuple(stencils))
