import numpy as np
from tools import Stencil
from algebra.core.space import Space
from discrete.fd.domain import FDDomain
from .lap_stencil_operator import LapStencilOperator
from .div_stencil_operator import DivStencilOperator
from .grad_stencil_operator import GradStencilOperator

def eye(space: Space[FDDomain], components: int) -> LapStencilOperator:
    contribs = {}
    grid = space.domain.grid
    # TODO: verify what should we really do in such situation
    # for ax in range(len(space.shape)):
    #     central_stencil = {0: 1}
    #     contribs[ax] = central_stencil
    central_stencil = {0: 1}
    contribs[0] = central_stencil
    stencil = Stencil(contribs)
    return LapStencilOperator(space, components, stencil)

def laplace(space: Space[FDDomain], components: int) -> LapStencilOperator:
    contribs = {}
    grid = space.domain.grid
    for ax in range(len(space.shape)):
        dh = grid.ax_spacing(ax)
        central_stencil = {
            -1: 1 / dh**2,
            0:  -2 / dh**2,
            1:  1 / dh**2
        }
        contribs[ax] = central_stencil
    stencil = Stencil(contribs)
    return LapStencilOperator(space, components, stencil)

def grad(space: Space[FDDomain], components: int) -> GradStencilOperator:
    central_contribs = {}
    grid = space.domain.grid
    for ax in range(space.ndim):
        dh = grid.ax_spacing(ax)
        central_stencil = {-1: -1 / (2*dh), 1:  1 / (2*dh)}
        central_contribs[ax] = central_stencil
    central_stencil = Stencil(central_contribs)
    operator = GradStencilOperator(space, components, central_stencil)
    for bid, boundary in space.domain.boundaries.items():
        dh = grid.ax_spacing(boundary.ax)
        inward_dir = boundary.inward_dir
        coeffs = boundary.side * (np.array([3, -4, 1]) / (2 * dh))
        boundary_stencil = {
            0: coeffs[0],
            inward_dir: coeffs[1],
            2*inward_dir: coeffs[2]
        }
        central_stencil = {-1: -1 / (2*dh), 1:  1 / (2*dh)}
        contribs = {ax: central_stencil.copy() for ax in range(space.ndim)}
        contribs[boundary.ax] = boundary_stencil
        stencil = Stencil(contribs)
        operator.boundary_stencils[bid] = stencil
    return operator

def div(space: Space[FDDomain], components: int) -> DivStencilOperator:
    central_contribs = {}
    grid = space.domain.grid
    for ax in range(space.ndim):
        dh = grid.ax_spacing(ax)
        central_stencil = {-1: -1 / (2*dh), 1:  1 / (2*dh)}
        central_contribs[ax] = central_stencil
    central_stencil = Stencil(central_contribs)
    operator = DivStencilOperator(space, components, central_stencil)
    for bid, boundary in space.domain.boundaries.items():
        dh = grid.ax_spacing(boundary.ax)
        inward_dir = boundary.inward_dir
        coeffs = boundary.side * ([3, -4, 1] / (2 * dh))
        boundary_stencil = {
            0: coeffs[0],
            inward_dir: coeffs[1],
            2*inward_dir: coeffs[2]
        }
        central_stencil = {-1: -1 / (2*dh), 1:  1 / (2*dh)}
        contribs = {ax: central_stencil.copy() for ax in range(space.ndim)}
        contribs[boundary.ax] = boundary_stencil
        stencil = Stencil(contribs)
        operator.boundary_stencils[bid] = stencil
    return operator
