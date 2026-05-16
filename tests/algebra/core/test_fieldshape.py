import numpy as np
from algebra.fieldshape import FieldShape


def test_fieldshape():
    shape = FieldShape((3, 10, 10), 2)
    assert shape.spacedim == 2
    assert shape.rank == 1
    assert shape.space == (10, 10)
    assert shape.ranks == (3,)


def test_fieldshaped_numpy_array():
    shape = FieldShape((3, 10, 10), 2)
    arr = np.zeros(shape)
    assert arr.shape == shape


def test_fieldshaped_compariosn():
    twodim = FieldShape((3, 10, 10), 2)
    onedim = FieldShape((3, 10, 10), 1)

    assert twodim != onedim

    other_twodim = FieldShape((3, 10, 10), 2)

    assert twodim == other_twodim
