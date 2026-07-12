from dataclasses import dataclass

from algebra.domain import bcs


@dataclass(frozen=True)
class FDBoundary(bcs.Boundary):
    ax: int
    side: int
    exclude_corners: bool
    dh: float
