from dataclasses import dataclass
import numpy as np
from algebra.systems.bcs import BoundaryCondition, BCType, BCTool

from tools import region
from .domain import FDDomain, FDBoundary
from .dx.space_stencil_operator import SpaceStencilOperator

@dataclass(frozen=True)
class FDBCondition():
    boundary: FDBoundary
    value: float

def apply_dirichlet(
    bc: FDBCondition,
    lhs: SpaceStencilOperator,
    rhs: np.ndarray
):
    stencil = lhs.boundary_stencils[bc.boundary.id]
    for ax in stencil.contribs.keys():
        if ax != bc.boundary.ax:
            stencil.contribs[ax] = {0: 0}

    field = np.zeros(lhs.input_shape)
    for comp in range(lhs.input_components):
        field[comp][bc.boundary.region] = bc.value
        dirichlet_contrib = np.zeros(lhs.space.shape)
        ax_range = stencil.ax_range(bc.boundary.ax, bc.boundary.inward_dir)
        offsets = [0 for ax in range(bc.boundary.grid.ndim)]
        ranges = [ax_range for i in range(bc.boundary.grid.ndim)]
        offsets[bc.boundary.ax] = tuple(ranges)
        boundary_interior = region.interior(bc.boundary.grid.shape, tuple(offsets))
        stencil.apply_to_region_on_ax(
            field[comp],
            dirichlet_contrib,
            boundary_interior,
            bc.boundary.ax
        )
        rhs[comp][bc.boundary.region] = 0.0
        rhs[comp] -= dirichlet_contrib
    stencil.contribs[bc.boundary.ax] = {0: 1}

def apply_neumann(
    bc: FDBCondition,
    lhs: SpaceStencilOperator,
    rhs: np.ndarray
):
    stencil = lhs.boundary_stencils[bc.boundary.id]
    contribs = stencil.contribs[bc.boundary.ax]
    h = lhs.domain.grid.ax_spacing(bc.boundary.ax)
    for offset, value in contribs.copy().items():
        if offset * bc.boundary.inward_dir < 0:
            contrib = value
            contribs[-offset] = contribs.get(-offset, 0.0) + contrib
            contribs.pop(offset)
            for comp in range(lhs.output_components):
                rhs[comp][bc.boundary.region] -= contrib * 2 * h * bc.value

class FDBCTool(BCTool):
    def __init__(self, domain: FDDomain):
        self._domain = domain

    def apply(
        self,
        bcs: list[BoundaryCondition],
        lhs: SpaceStencilOperator,
        rhs: np.ndarray
    ):
        for bc in bcs:
            boundary = self._domain.boundary(bc.boundary)
            fd_bc = FDBCondition(boundary, bc.value)
            if bc.bctype == BCType.DIRICHLET:
                apply_dirichlet(fd_bc, lhs, rhs)
            elif bc.bctype == BCType.NEUMANN:
                apply_neumann(fd_bc, lhs, rhs)

    def post_solve(self, bc: BoundaryCondition, field: np.ndarray):
        if bc.bctype == BCType.DIRICHLET:
            boundary = self._domain.boundary(bc.boundary)
            field[boundary.region] = bc.value
