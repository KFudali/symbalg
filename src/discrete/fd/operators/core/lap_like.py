import numpy as np
from algebra.space import Space, ShapeTransform
from discrete.fd.tools.stencil import AxStencil
from .fd_operator import FDOperator


class FDLapLikeOperator(FDOperator):
    """Laplace-like operator. Does not modify rank of field.

    A field with shape ``(*outer, space_ndim, *space)`` is mapped to a field
    of the same shape .
    """

    def __init__(self, space: Space, stencils: tuple[AxStencil, ...]):
        super().__init__(space, ShapeTransform.NONE, stencils)

    def apply(self, inp: np.ndarray, out: np.ndarray):
        field_rank = len(inp.shape[: -self.space.ndim])
        if field_rank == 0:
            for ax in range(self.space.ndim):
                self.stencils[ax].eval_to(ax, inp, out)
        else:
            for comp in range(inp.shape[0]):
                self.apply(inp[comp], out[comp])
