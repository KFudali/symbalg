from dataclasses import dataclass

from discrete.core.domain import Boundary


@dataclass(frozen=True)
class FDBoundary(Boundary):
    ax: int
    side: int
    exclude_corners: bool
    dh: float
