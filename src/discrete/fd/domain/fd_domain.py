from tools.geometry import StructuredGridND
from discrete.core.domain import Domain, BoundaryId
from .fd_boundary import FDBoundary


class FDDomain(Domain):
    def __init__(self, grid: StructuredGridND):
        super().__init__()
        self._grid = grid
        self._boundaries = dict[BoundaryId, FDBoundary]()
        self._boundaries_by_ax = dict[int, tuple[BoundaryId, BoundaryId]]()
        self._mark_boundaries()

    @property
    def grid(self) -> StructuredGridND:
        return self._grid

    @property
    def boundaries(self) -> dict[BoundaryId, FDBoundary]:
        return self._boundaries.copy()

    def boundary(self, boundary_id: BoundaryId) -> FDBoundary:
        return self._boundaries[boundary_id]

    def ax_boundaries(self, ax: int) -> tuple[BoundaryId, BoundaryId]:
        return self._boundaries_by_ax[ax]

    def _mark_boundaries(self):
        next_id = 0
        for ax in range(self.grid.ndim):
            left_id = BoundaryId(next_id)
            right_id = BoundaryId(next_id + 1)
            exclude_conrers = ax % 2 != 0
            dh = self._grid.ax_spacing(ax)
            left = FDBoundary(left_id, ax, -1, exclude_conrers, dh)
            right = FDBoundary(right_id, ax, 1, exclude_conrers, dh)
            self._boundaries[left_id] = left
            self._boundaries[right_id] = right
            self._boundaries_by_ax[ax] = (left_id, right_id)
            next_id += 2
