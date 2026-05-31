from dataclasses import dataclass
import numpy as np
from tools.geometry import StructuredGridND

from discrete.core import Discretization
from algebra.space import Space
from .domain import FDDomain
from .dt_operators import FDDtOperators
from .dx_operators import FDDxOperators
from .bc_tool import FDBCTool


class FdDiscretization(Discretization[FDDomain]):
    def __init__(self, grid: StructuredGridND):
        domain = FDDomain(grid)
        space = Space(grid.shape)
        super().__init__(space, domain)
        self._dx = FDDxOperators(space, grid)
        self._dt = FDDtOperators(space, self._time)
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
            linspace = np.linspace(0, (nx - 1) * dx, nx)
            spaces.append(linspace)
        return np.meshgrid(*tuple(spaces))
