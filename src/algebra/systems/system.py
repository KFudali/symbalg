from abc import ABC, abstractmethod
from algebra.expression import Expression

class System(ABC):
    @abstractmethod
    def solve(self) -> Expression:
        pass
