from abc import ABC, abstractmethod
from algebra.core.expression import Expression

class System(ABC):
    @abstractmethod
    def solve(self) -> Expression:
        pass
