from __future__ import annotations
import numpy as np
from algebra.core.space import Space

from tools import Stencil, region
from ..space_stencil_operator import SpaceStencilOperator
from ..domain import FDDomain


class GradStencilOperator(SpaceStencilOperator):
    def __init__(
        self,
        space: Space[FDDomain],
        input_components: int,
        stencil: Stencil,
    ):
        ndim = len(space.shape)
        super().__init__(space, input_components, input_components * ndim, stencil)

    def _new(
        self,
        interior: Stencil,
        boundary_stencils: dict,
    ) -> "GradStencilOperator":
        new = GradStencilOperator(self.space, self.input_components, interior)
        for bid, stencil in boundary_stencils.items():
            new.boundary_stencils[bid] = stencil.copy()
        return new

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        for component in range(self.output_components):
            self._apply_to_component(input_field, output_field, component)

    def _apply_to_component(
        self,
        input: np.ndarray,
        output: np.ndarray,
        output_component: int,
    ):
        input_idx = output_component // len(self.space.shape)
        spatial_dim = output_component % len(self.space.shape)

        input_arr = input[input_idx]
        output_arr = output[output_component]

        interior = region.interior(
            input_arr.shape,
            tuple(self.interior_stencil.ax_ranges().values()),
        )
        self.interior_stencil.apply_to_region_on_ax(
            input_arr, output_arr, interior, spatial_dim
        )
        for bid, stencil in self.boundary_stencils.items():
            boundary = self.space.domain.boundary(bid)
            stencil.apply_to_region_on_ax(
                input_arr, output_arr, boundary.region, spatial_dim
            )
