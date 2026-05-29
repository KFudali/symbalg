import numpy as np
from tools.stencil import AxStencil
from algebra.core import FieldShape
from discrete.fd.fd_operator import FDOperator
from .ders import dx


class FDGradLikeOperator(FDOperator):
    """Gradient-like operator. Increases tensor rank by 1.

    A field with shape ``(*ranks, *space)`` is mapped to a field with shape
    ``(*ranks, self.ndim, *space)`` — a new component axis (size
    ``self.ndim``) is inserted just before the spatial axes.
    """

    def __init__(self, stencils: tuple[AxStencil, ...], input_shape: FieldShape):
        output_shape = FieldShape(
            (input_shape.spacedim, *input_shape), input_shape.spacedim
        )
        super().__init__(input_shape, output_shape, stencils)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        field_rank = len(input_field.shape[: -self.input_shape.spacedim])
        if field_rank == 0:
            for ax in range(self.input_shape.spacedim):
                self.stencils[ax].eval_to(ax, input_field, output_field[ax])
        else:
            for comp in range(input_field.shape[0]):
                self.apply(input_field[comp], output_field[comp])


def grad(order: int, h: float, input_shape: FieldShape) -> "FDGradLikeOperator":
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(input_shape.spacedim))
    return FDGradLikeOperator(stencils, input_shape)
