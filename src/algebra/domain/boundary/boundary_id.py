from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryId:
    key: int

    def is_valid(self) -> bool:
        return self.key >= 0
