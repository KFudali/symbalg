from __future__ import annotations
import numpy as np
from typing import Self
from algebra.core.space import Space

from tools import Stencil, region
from algebra.core.space.domain import BoundaryId
from .space_stencil_operator import SpaceStencilOperator
from ..domain import FDDomain


class LapStencilOperator(SpaceStencilOperator):
    def __init__(
        self,
        space: Space[FDDomain],
        components: int,
        stencil: Stencil
    ):
        super().__init__(space, components, components, stencil)

    def _new(
        self, interior: Stencil, boundary_stencils: dict[BoundaryId, Stencil],
    ) -> Self:
        new = LapStencilOperator(
            self.space, 
            self.input_components, 
            interior.copy()
        )
        for bid, stencil in boundary_stencils.items():
            new.boundary_stencils[bid] = stencil.copy()
        return new

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        for component in range(self.input_components):
            self._apply_to_component(
                input_field[component],
                output_field[component]
            )

    def _apply_to_component(
        self,
        input: np.ndarray,
        output: np.ndarray
    ):
        interior = region.interior(
            input.shape,
            tuple(self.interior_stencil.ax_ranges().values()),
        )
        self.interior_stencil.apply_to_region(input, output, interior)
        for bid, stencil in self.boundary_stencils.items():
            boundary = self.space.domain.boundary(bid)
            stencil.apply_to_region(input, output, boundary.region)
