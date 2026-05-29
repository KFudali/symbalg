import numpy as np
from tools.stencil import AxStencil
from algebra.core import FieldShape
from discrete.fd.fd_operator import FDOperator
from .ders import ddx


class FDLapLikeOperator(FDOperator):
    """Laplace-like operator. Does not modify rank of field.

    A field with shape ``(*outer, space_ndim, *space)`` is mapped to a field
    of the same shape .
    """

    def __init__(self, stencils: tuple[AxStencil], input_shape: FieldShape):
        super().__init__(input_shape, input_shape, stencils)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        field_rank = len(input_field.shape[: -self.input_shape.spacedim])
        if field_rank == 0:
            for ax in range(self.input_shape.ndim):
                self.stencils[ax].eval_to(ax, input_field, output_field)
        else:
            for comp in range(input_field.shape[0]):
                self._apply(input_field[comp], output_field[comp])


def laplace(order: int, h: float, input_shape: FieldShape) -> FDLapLikeOperator:
    if ddx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(ddx.stencil(order, h) for _ in range(input_shape.spacedim))
    return FDLapLikeOperator(stencils, input_shape)
