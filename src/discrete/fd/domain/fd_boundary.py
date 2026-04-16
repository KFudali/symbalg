from dataclasses import dataclass
from algebra.core.space.domain import Boundary
from tools import Region
from tools.geometry import StructuredGridND

@dataclass(frozen=True)
class FDBoundary(Boundary):
    grid: StructuredGridND
    ax: int
    side: int

    @property
    def region(self) -> Region:
        return self.grid.boundary(self.ax, self.side)

    @property
    def inward_spacing(self) -> float:
        return self.grid.ax_spacing(self.ax)
