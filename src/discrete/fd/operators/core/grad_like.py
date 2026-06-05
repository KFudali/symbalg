import numpy as np
from algebra.space import Space, ShapeTransform
from discrete.fd.tools.stencil import AxStencil
from .fd_operator import FDOperator


class FDGradLikeOperator(FDOperator):
    """Gradient-like operator. Increases tensor rank by 1.

    A field with shape ``(*ranks, *space)`` is mapped to a field with shape
    ``(*ranks, self.ndim, *space)`` — a new component axis (size
    ``self.ndim``) is inserted just before the spatial axes.
    """

    def __init__(self, space: Space, stencils: tuple[AxStencil, ...]):
        super().__init__(space, ShapeTransform.INCREASE_RANK, stencils)

    def apply(self, inp: np.ndarray, out: np.ndarray):
        field_rank = len(inp.shape[: -self.space.ndim])
        if field_rank == 0:
            for ax in range(self.space.ndim):
                self.stencils[ax].eval_to(ax, inp, out[ax])
        else:
            for comp in range(out.shape[0]):
                self.apply(inp[comp], out[comp])
