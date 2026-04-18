import numpy as np
from algebra.core.space import SpaceObject

from discrete import DiscreteSpace
from discrete.field import DiscreteField, DiscreteFieldValueBuffer
from tools.buffer import DequeValueBuffer
from .time import TimeSeries

class FieldSpace(SpaceObject[DiscreteSpace]):
    def __init__(self, space: DiscreteSpace):
        super().__init__(space)
        self._time_series = TimeSeries(space.time)

    def field(self, components: int, init_value: float = 0.0) -> DiscreteField:
        shape = (components, *self.space.shape)
        buffer = DequeValueBuffer(shape)
        buffer.set(init_value * np.ones(shape = shape, dtype=float))
        field_buffer = DiscreteFieldValueBuffer(self.space, components, buffer)
        field = DiscreteField(field_buffer)
        return field

    @property
    def time(self) -> TimeSeries:
        return self._time_series
