from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Space:
    shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)
