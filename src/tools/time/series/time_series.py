from abc import abstractmethod, ABC
import numpy as np

class TimeDim(ABC):
    @abstractmethod
    def dt(self) -> float:
        pass

    @abstractmethod
    def now(self) -> float:
        pass

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
