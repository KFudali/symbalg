from .space import TSpace
from .space_object import SpaceObject

class FieldShaped(SpaceObject[TSpace]):
    def __init__(self, space: TSpace, components: int):
        super().__init__(space)
        self._components = components

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._components, *self.space.shape)

    @property
    def components(self) -> int:
        return self._components
