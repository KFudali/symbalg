class TimeDim():
    def __init__(self):
        self._current = 0.0

    @property
    def current(self) -> float:
        return self._current
