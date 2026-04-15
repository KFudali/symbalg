from abc import abstractmethod
import numpy as np
from algebra.space import TimeDim

class TimeSeries(TimeDim):
    @abstractmethod
    def advance(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def domain(self) -> np.ndarray:
        pass
