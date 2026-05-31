import numpy as np

from discrete.core import Discretization

from discrete import DiscreteSpace
from tools.buffer import DequeValueBuffer
from .time_series import TimeSeries


class FieldSpace:
    def __init__(self, discretization: Discretization):
        super().__init__()
        self._discretization = discretization
        self._time_series = TimeSeries(discretization.time)

    def field(self, 0) -> Field:
        shape = FieldShape((*ranks, self._space.shape), self._space.ndim)
        buffer = DequeValueBuffer(shape)
        buffer.set(init_value * np.ones(shape=shape, dtype=float))
        field = Field(buffer)
        return field

    def dx(self):
        pass

    def dt(self):
        pass

    @property
    def time(self) -> TimeSeries:
        return self._time_series
