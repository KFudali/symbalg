import numpy as np
from tools.geometry import StructuredGridND

from discrete.core import Discretization
from .domain import FDDomain
from .dt_operators import FDDtOperators
from .dx_operators import FDDxOperators


class FdDiscretization(Discretization[FDDomain]):
    def __init__(self, grid: StructuredGridND):
        domain = FDDomain(grid)
        super().__init__(domain)
        self._dx = FDDxOperators(domain)
        self._dt = FDDtOperators(domain, self._time)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.domain.grid.shape

    @property
    def dt(self) -> FDDtOperators:
        return self._dt

    @property
    def dx(self) -> FDDxOperators:
        return self._dx

    def points(self) -> tuple[np.ndarray, ...]:
        grid = self.domain.grid
        spaces = []
        for nx, dx in zip(grid.shape, grid.spacing):
            linspace = np.linspace(0, (nx - 1) * dx, nx)
            spaces.append(linspace)
        return np.meshgrid(*tuple(spaces))
