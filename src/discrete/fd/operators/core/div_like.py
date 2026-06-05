import numpy as np
from algebra.space import Space, ShapeTransform
from discrete.fd.tools.stencil import AxStencil
from .fd_operator import FDOperator


class FDDivLikeOperator(FDOperator):
    """Divergence-like operator. Decreases tensor rank by 1.

    A field with shape ``(*outer, self.ndim, *space)`` is mapped to a field
    with shape ``(*outer, *space)``. The contracted axis is the **last** rank
    axis (the one immediately preceding the spatial axes), which must have
    size ``self.ndim``.
    """

    def __init__(self, space: Space, stencils: tuple[AxStencil, ...]):
        super().__init__(space, ShapeTransform.REDUCE_RANK, stencils)

    def apply(self, inp: np.ndarray, out: np.ndarray):
        rank = len(inp.shape[: -self.space.ndim])
        assert rank >= 1, "DivLikeOperator requires field rank >= 1"
        assert (
            inp.shape[rank - 1] == self.space.ndim
        ), "DivLikeOperator requires the last rank axis to match space.ndim"
        assert out.shape == inp.shape[: rank - 1] + inp.shape[rank:]
        if rank == 1:
            for ax in range(self.space.ndim):
                self.stencils[ax].eval_to(ax, inp[ax], out)
        else:
            assert inp.shape[0] == out.shape[0], (
                "DivLikeOperator requires field and out to share leading " "rank dims"
            )
            for comp in range(inp.shape[0]):
                self.apply(inp[comp], out[comp])
