from __future__ import annotations
import numpy as np
from algebra.core.space import Space

from tools import Stencil, region
from ..space_stencil_operator import SpaceStencilOperator
from ..domain import FDDomain


class DivStencilOperator(SpaceStencilOperator):
    def __init__(
        self,
        space: Space[FDDomain],
        input_components: int,
        stencil: Stencil,
    ):
        ndim = len(space.shape)
        super().__init__(space, input_components, input_components // ndim, stencil)

    def _new(
        self,
        interior: Stencil,
        boundary_stencils: dict,
    ) -> "DivStencilOperator":
        new = DivStencilOperator(self.space, self.input_components, interior)
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
        ndim = len(self.space.shape)
        output_arr = output[output_component]

        interior = region.interior(
            input[0].shape,
            tuple(self.interior_stencil.ax_ranges().values()),
        )

        for spatial_dim in range(ndim):
            input_idx = output_component * ndim + spatial_dim
            input_arr = input[input_idx]
            self.interior_stencil.apply_to_region_on_ax(
                input_arr, output_arr, interior, spatial_dim
            )

        for bid, stencil in self.boundary_stencils.items():
            boundary = self.space.domain.boundary(bid)
            for spatial_dim in range(ndim):
                input_idx = output_component * ndim + spatial_dim
                input_arr = input[input_idx]
                stencil.apply_to_region_on_ax(
                    input_arr, output_arr, boundary.region, spatial_dim
                )
