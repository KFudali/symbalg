import numpy as np
from .time_series import TimeSeries

class AdaptiveDtTimeSeries(TimeSeries):
    def __init__(self, first_dt: float, init_time: float = 0.0):
        self._dt = first_dt
        self._time = [init_time]
        self._dts = [first_dt]

    def dt(self) -> float:
        return self._dt

    def time(self) -> float:
        return self._time[-1]

    def set_next_dt(self, dt: float):
        self._dt = dt

    def advance(self):
        self._time.append(self.time + self._dt)
        self._dts.append(self._dt)

    def reset(self):
        self._time = [0.0]
        self._dts = [self._dts[-1]]

    def domain(self) -> np.ndarray:
        return np.array(self._time)

    def dts(self) -> np.ndarray:
        return np.array(self._dts)
