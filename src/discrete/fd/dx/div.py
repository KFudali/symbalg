import numpy as np
from discrete.fd.fd_operator import FDOperator
from algebra.core import FieldShape
from tools.stencil import AxStencil
from .ders import dx


class FDDivLikeOperator(FDOperator):
    """Divergence-like operator. Decreases tensor rank by 1.

    A field with shape ``(*outer, self.ndim, *space)`` is mapped to a field
    with shape ``(*outer, *space)``. The contracted axis is the **last** rank
    axis (the one immediately preceding the spatial axes), which must have
    size ``self.ndim``.
    """

    def __init__(self, stencils: tuple[AxStencil, ...], input_shape: FieldShape):
        assert input_shape.rank >= 1, "DivLikeOperator requires field rank >= 1"
        if input_shape.rank == 1:
            assert input_shape[0] == input_shape.spacedim, (
                "DivLikeOperator base case requires the leading axis to "
                "match spacedim"
            )
        output_shape = FieldShape(
            (*input_shape[: input_shape.rank - 1], *input_shape[input_shape.rank :]),
            input_shape.spacedim,
        )
        super().__init__(input_shape, output_shape, stencils)

    def _apply(self, input_field: np.ndarray, output_field: np.ndarray):
        field_rank = len(input_field.shape[: -self.input_shape.spacedim])
        if field_rank == 1:
            assert input_field.shape[0] == self.input_shape.spacedim, (
                "DivLikeOperator base case requires the leading axis to "
                "match self.ndim"
            )
            assert output_field.shape == input_field.shape[1:], (
                "DivLikeOperator base case requires out.shape == " "field.shape[1:]"
            )
            for ax in range(self.input_shape.spacedim):
                self.stencils[ax].eval_to(ax, input_field[ax], output_field)
        else:
            assert input_field.shape[0] == output_field.shape[0], (
                "DivLikeOperator requires field and out to share leading " "rank dims"
            )
            for comp in range(input_field.shape[0]):
                self.apply(input_field[comp], output_field[comp])


def div(order: int, h: float, input_field: FieldShape) -> DivLikeOperator:
    if dx.stencil(order, h) is NotImplemented:
        return NotImplemented
    stencils = tuple(dx.stencil(order, h) for _ in range(input_field.spacedim))
    return DivLikeOperator(stencils, input_field)
