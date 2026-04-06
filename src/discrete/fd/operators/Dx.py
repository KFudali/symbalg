from .lap_stencil_operator import LapStencilOperator
from .div_stencil_operator import DivStencilOperator
from .grad_stencil_operator import GradStencilOperator

from algebra.space import Space
from ..domain import FDDomain
from tools import Stencil


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

def grad(space: Space[FDDomain], components: int) -> GradStencilOperator: pass

def div(space: Space[FDDomain], components: int) -> DivStencilOperator: pass