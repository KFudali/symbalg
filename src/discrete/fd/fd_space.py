from discrete.core import DiscreteSpace
from algebra.core.space import TimeDim
from .domain import FDDomain

from .dt_operators import FDDtOperators
from .dx_operators import FDDxOperators
from .bc_tool import FDBCTool
from tools.geometry import StructuredGridND

class FdDiscreteSpace(DiscreteSpace[FDDomain]):
    def __init__(self, grid: StructuredGridND, time: TimeDim):
        domain = FDDomain(grid)
        super().__init__(domain, time)
        self._dt = FDDtOperators()
        self._dx = FDDxOperators(space = self)
        self._bcs = FDBCTool(space = self)

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
