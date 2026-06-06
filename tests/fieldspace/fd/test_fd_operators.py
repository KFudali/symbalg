import pytest
from fieldspace import FieldSpace
from tools.geometry import StructuredGridND
from discrete import fd


@pytest.fixture
def fd_space():
    grid = StructuredGridND((10, 10), (0.01, 0.01))
    discr = fd.FdDiscretization(grid)
    return FieldSpace(discr)
