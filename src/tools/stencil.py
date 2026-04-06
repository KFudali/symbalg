import numpy as np
from tools.region import Region
from .stencil_operators_mixin import StencilOperatorsMixin

class Stencil(StencilOperatorsMixin):
    def __init__(self, contribs: dict[int, dict[int, float]]):
        super().__init__()
        self.contribs = contribs

    def copy(self):
        return Stencil({
            ax: dict(offsets)
            for ax, offsets in self.contribs.items()
        })

    @property
    def ndim(self) -> int:
        return len(self.contribs)

    def central(self) -> float:
        return sum(c.get(0, 0.0) for c in self.contribs.values())

    def on_ax(self, ax: int, dir: int | None = None) -> dict[int, float]:
        ax_contribs = self.contribs[ax]
        if dir is None:
            return ax_contribs
        if dir > 0:
            return {p: v for p, v in ax_contribs.items() if p > 0}
        if dir < 0:
            return {p: v for p, v in ax_contribs.items() if p < 0}
        raise ValueError("dir must be None, positive, or negative")

    def ax_range(self, ax: int, dir: int) -> int:
        ax_contribs = self.contribs[ax]
        if dir > 0:
            positives = [p for p in ax_contribs if p > 0]
            return max(positives, default=0)
        if dir < 0:
            negatives = [p for p in ax_contribs if p < 0]
            return min(negatives, default=0)
        raise ValueError("dir must be positive or negative")

    def ax_ranges(self) -> dict[int, tuple[int, int]]:
        return {ax: (self.ax_range(ax, -1), self.ax_range(ax, 1)) for ax in range(self.ndim)}

    def apply_to_region(
        self, field: np.ndarray, out: np.ndarray, region: Region
    ):
        for ax in range(field.ndim):
            self.apply_to_region_on_ax(field, out, region, ax)

    def apply_to_region_on_ax(
        self, field: np.ndarray, out: np.ndarray, region: Region, ax: int
    ):
        contrib = self.on_ax(ax)
        for k, c in contrib.items():
            contrib_region = region.shift(ax, k)
            out[region] += c * field[contrib_region]
