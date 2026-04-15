from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Self
import numpy as np

from algebra.operator import SpaceOperator
from algebra.space import Space
from algebra.space.domain import BoundaryId

from tools import Stencil
from .domain import FDDomain

class SpaceStencilOperator(SpaceOperator[FDDomain], ABC):
    def __init__(
        self,
        space: Space[FDDomain],
        input_components: int,
        output_components: int,
        stencil: Stencil
    ):
        super().__init__(space, input_components, output_components)
        self._interior_stencil = stencil
        self._boundary_stencils = dict[BoundaryId, Stencil]()
        for boundary_id in space.domain.boundaries.keys():
            self._boundary_stencils[boundary_id] = stencil.copy()

    def _new(
        self, interior: Stencil, boundary_stencils: dict[BoundaryId, Stencil],
    ) -> Self:
        new = SpaceStencilOperator(
            self.space,
            self.input_components,
            self.output_components,
            interior.copy()
        )
        for bid, stencil in boundary_stencils.items():
            new.boundary_stencils[bid] = stencil.copy()

    @abstractmethod
    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        pass

    @property
    def domain(self) -> FDDomain:
        return self.space.domain

    @property
    def interior_stencil(self) -> Stencil:
        return self._interior_stencil

    @property
    def boundary_stencils(self) -> dict[BoundaryId, Stencil]:
        return self._boundary_stencils

    @property
    def stencils(self) -> list[Stencil]:
        return list(self.boundary_stencils.values()) + [self.interior_stencil]

    def copy(self) -> "SpaceStencilOperator":
        boundaries = {
            bid: stencil.copy() for bid, stencil in self.boundary_stencils.items()
        }
        return self._new(self._interior_stencil.copy(), boundaries)

    def __neg__(self) -> "SpaceStencilOperator":
        return self._new(
            -self.interior_stencil,
            {bid: -st for bid, st in self.boundary_stencils.items()},
        )

    def add(self, other: Self) -> Self:
        return self._new(
            (self.interior_stencil + other.interior_stencil),
            {
                bid: (stencil + other.boundary_stencils[bid])
                for bid, stencil in self.boundary_stencils.items()
            },
        )

    def mul(self, other: Self) -> Self:
        return self._new(
            (self.interior_stencil * other.interior_stencil),
            {
                bid: (stencil * other.boundary_stencils[bid])
                for bid, stencil in self.boundary_stencils.items()
            },
        )

    def scale(self, other: float) -> Self:
        return self._new(
            (self.interior_stencil * other),
            {
                bid: (stencil * other)
                for bid, stencil in self.boundary_stencils.items()
            },
        )

    def scale_arr(self, other: np.ndarray) -> Self:
        return NotImplemented
