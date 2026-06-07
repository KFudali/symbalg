import pytest
import numpy as np
from discrete.fd.tools.stencil import AxStencil, Stencil
from discrete.fd.operators import dx
from algebra.space import Space
from algebra.field import Field, FieldShape
from tools.buffer import DequeValueBuffer

space = Space((10, 10))
one = Stencil({0: 1.0})
two = Stencil({0: 2.0})


def test_lap_apply_to_field():
    ax = AxStencil(two.copy(), (), ())
    op = dx.FDLapLikeOperator(space, (ax, ax.copy()))

    fieldshape = FieldShape(space, (1,))
    buffer = DequeValueBuffer(fieldshape.shape)
    buffer.set(np.ones(shape=fieldshape.shape, dtype=float))
    field = Field(fieldshape, buffer)

    out = op.of(field).eval()
    assert np.array_equal(4.0 * field.value().eval(), out)
