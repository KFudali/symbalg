from discrete import DiscreteTime

class TimeSeries():
    def __init__(self, time: DiscreteTime):
        self._discrete_time = time

    def run(self, duration: float, init_dt: float = 0.01):
        to_go = duration
        self._discrete_time.set_dt(init_dt)
        while to_go >= 0:
            yield self._discrete_time.current
            to_go -= self._discrete_time.dt()
            self._discrete_time.advance()

    def adapt_dt(self, dt: float):
        self._discrete_time.set_dt(dt)

    def reset(self):
        self._discrete_time.reset()
