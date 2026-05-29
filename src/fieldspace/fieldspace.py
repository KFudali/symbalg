import numpy as np
from algebra.core import FieldShape

from discrete import DiscreteSpace
from discrete.field import DiscreteField, DiscreteFieldValueBuffer
from tools.buffer import DequeValueBuffer
from .time import TimeSeries


class FieldSpace(SpaceObject[DiscreteSpace]):
    def __init__(self, space: DiscreteSpace):
        super().__init__(space)
        self._time_series = TimeSeries(space.time)

    def field(self, ranks: tuple[int, ...], init_value: float = 0.0) -> DiscreteField:
        shape = FieldShape((*ranks, self._space.shape), self._space.ndim)
        buffer = DequeValueBuffer(shape)
        buffer.set(init_value * np.ones(shape=shape, dtype=float))
        field = DiscreteField(buffer)
        return field

    @property
    def time(self) -> TimeSeries:
        return self._time_series
