from __future__ import annotations


class FieldShape(tuple):
    def __new__(cls, shape: tuple[int, ...], spacedim: int):
        shape = tuple(shape)
        assert (
            len(shape) >= spacedim
        ), "space ndim has to be less or equal to total shape ndim"
        instance = super().__new__(cls, shape)
        instance._spacedim = spacedim
        return instance

    def __repr__(self) -> str:
        return f"FieldShape({tuple(self)!r}, spacedim={self.spacedim})"

    def __eq__(self, other) -> bool:
        if isinstance(other, FieldShape):
            return tuple(self) == tuple(other) and self._spacedim == other._spacedim
        return tuple(self) == other

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((tuple(self), self._spacedim))

    def __reduce__(self):
        return (self.__class__, (tuple(self), self._spacedim))

    @property
    def ndim(self) -> int:
        return len(self)

    @property
    def spacedim(self) -> int:
        return self._spacedim

    @property
    def rank(self) -> int:
        return self.ndim - self.spacedim

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(self[: self.rank])

    @property
    def space(self) -> tuple[int, ...]:
        return tuple(self[self.rank :])

    def is_scalar(self) -> bool:
        return len(self) == 0

    @classmethod
    def scalar(cls) -> "FieldShape":
        return cls((), 0)


class FieldShaped:
    def __init__(self, shape: FieldShape):
        self._shape = shape

    @property
    def shape(self) -> FieldShape:
        return self._shape
