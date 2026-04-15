import numpy as np
from .time_series import TimeSeries

class ConstDtTimeSeries(TimeSeries):
    def __init__(self, dt: float, init_time: float = 0.0):
        self._dt = dt
        self._init_time = init_time
        self._time = init_time

    def dt(self) -> float:
        return self._dt

    def now(self) -> float:
        return self._time

    def advance(self):
        self._time += self._dt

    def reset(self):
        self._time = 0.0

    def domain(self) -> np.ndarray:
        return np.arange(
            self._init_time,
            self._time + self._dt,
            step=self._dt,
            dtype = float
        )
