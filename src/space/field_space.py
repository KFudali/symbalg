from algebra.space import Space
from .field import Field

class FieldSpace():
    def __init__(self, space: Space):
        self._space = space

    @property
    def space(self) -> Space:
        return self._space

    def field(self, components: int) -> Field:
        pass

    def scalar_field(self) -> Field:
        return self.field(components=1)

    def vector_field(self) -> Field:
        return self.field(components=self.space.ndim)
