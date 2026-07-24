from abc import ABC, abstractmethod
from algebra.field import Field
from algebra.domain.operator import AffineDomainOperator


class DtOperators(ABC):
    @abstractmethod
    def explicit(self, field: Field, order: int = 1) -> AffineDomainOperator: ...

    @abstractmethod
    def implicit(self, field: Field, order: int = 1) -> AffineDomainOperator: ...
