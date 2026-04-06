from algebra.space.domain import Domain, BoundaryId
from tools.geometry import StructuredGridND
from .fd_boundary import FDBoundary

class FDDomain(Domain):
    def __init__(self, grid: StructuredGridND):
        super().__init__()
        self._grid = grid
        self._boundaries = dict[BoundaryId, FDBoundary]()
        self._mark_boundaries()

    @property
    def grid(self) -> StructuredGridND:
        return self.grid

    @property
    def boundaries(self) -> dict[BoundaryId, FDBoundary]:
        return self._boundaries.copy()

    def boundary(self, boundary_id: BoundaryId) -> FDBoundary:
        return self._boundaries[boundary_id]

    def _mark_boundaries(self):
        next_id = 0
        for ax in self.grid.ndim:
            left_id = BoundaryId(next_id)
            right_id = BoundaryId(next_id + 1)
            left = FDBoundary(left_id, self.grid, ax, side=-1)
            right = FDBoundary(right_id, self.grid, ax, side=1)
            self._boundaries[left_id] = left
            self._boundaries[right_id] = right
