from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from algebra.exceptions import ShapeMismatchError
from algebra.expression.expression import Expression
from algebra.operator import SpaceOperator, Operator
from algebra.space import Space
from algebra.space.domain import BoundaryId

from tools import Stencil
from ..domain import FDDomain


class SpaceStencilOperator(SpaceOperator, ABC):
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

    @abstractmethod
    def _new(
        self, interior: Stencil, boundary_stencils: dict[BoundaryId, Stencil],
    ) -> "SpaceStencilOperator":
        pass

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

    def __add__(self, other: Operator | Expression | float) -> "SpaceStencilOperator":
        if not isinstance(other, SpaceStencilOperator):
            return NotImplemented
        if (
            self.input_shape != other.input_shape
            or self.output_shape != other.output_shape
        ):
            raise ShapeMismatchError("Operator shape mismatch")
        return self._new(
            self.interior_stencil + other.interior_stencil,
            {
                bid: self.boundary_stencils[bid]
                + other.boundary_stencils[bid]
                for bid in self.boundary_stencils
            }
        )

    def __sub__(self, other: Operator | Expression | float) -> "SpaceStencilOperator":
        if not isinstance(other, SpaceStencilOperator):
            return NotImplemented
        if (
            self.input_shape != other.input_shape
            or self.output_shape != other.output_shape
        ):
            raise ShapeMismatchError("Operator shape mismatch")
        return self._new(
            self.interior_stencil - other.interior_stencil,
            {
                bid: self.boundary_stencils[bid]
                - other.boundary_stencils[bid]
                for bid in self.boundary_stencils
            }
        )

    def __mul__(self, other: Operator | Expression | float) -> "SpaceStencilOperator":
        if isinstance(other, float):
            return self._new(
                self.interior_stencil,
                self.boundary_stencils,
            )
        return NotImplemented

    def __rmul__(self, other: float) -> "SpaceStencilOperator":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "SpaceStencilOperator":
        if isinstance(other, float):
            return self._new(
                self.interior_stencil,
                self.boundary_stencils,
            )
        return NotImplemented
