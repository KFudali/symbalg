from discrete.core import Discretization, DxOperators, DtOperators
from .fields import FieldFactory
from .systems import SystemFactory
from .monitors import MonitorFactory

from .time_series import TimeSeries


class FieldSpace:
    def __init__(self, discretization: Discretization):
        super().__init__()
        self._discrete = discretization
        self._systems = SystemFactory(discretization)
        self._fields = FieldFactory(discretization.space)
        self._monitors = MonitorFactory(discretization)
        self._time_series = TimeSeries(discretization.time)

    @property
    def dx(self) -> DxOperators:
        return self._discrete.dx

    @property
    def dt(self) -> DtOperators:
        return self._discrete.dt

    @property
    def systems(self) -> SystemFactory:
        return self._systems

    @property
    def fields(self) -> FieldFactory:
        return self._fields

    @property
    def monitors(self) -> MonitorFactory:
        return self._monitors

    @property
    def time(self) -> TimeSeries:
        return self._time_series
