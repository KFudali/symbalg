import numpy as np

from algebra.space import Space
from discrete.fd.tools.stencil import AxStencil


def as_array(space: Space, stencils: tuple[AxStencil, ...]) -> np.ndarray:
    assert space.ndim == len(stencils)
