import numpy as np
from tools.geometry import StructuredGridND
from tools.time import TimeDim

from discrete.core import DiscreteSpace

from .domain import FDDomain
from .dt_operators import FDDtOperators
from .dx_operators import FDDxOperators
from .bc_tool import FDBCTool

class FdDiscreteSpace(DiscreteSpace[FDDomain]):
    def __init__(self, grid: StructuredGridND):
        domain = FDDomain(grid)
        super().__init__(domain)
        self._dt = FDDtOperators()
        self._dx = FDDxOperators(space = self)
        self._bcs = FDBCTool(domain)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.domain.grid.shape

    @property
    def bcs(self) -> FDBCTool:
        return self._bcs

    @property
    def dt(self) -> FDDtOperators:
        return self._dt

    @property
    def dx(self) -> FDDxOperators:
        return self._dx

    def points(self) -> np.ndarray:
        grid = self.domain.grid
        spaces = []
        for nx, dx in zip(grid.shape, grid.spacing):
            linspace = np.linspace(0, (nx-1) * dx, nx)
            spaces.append(linspace)
        return np.meshgrid(*tuple(spaces))

