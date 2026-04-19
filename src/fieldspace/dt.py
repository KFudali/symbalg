from algebra.field import FieldOperator
from discrete import DiscreteField, DiscreteSpace

def euler(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    dt = field.space.dt.euler(field, field.space.time)
    return FieldOperator(field, dt.operator, dt.expression)

def bfd(field: DiscreteField, order: int = 1) -> FieldOperator[DiscreteSpace]:
    dt = field.space.dt.bfd(field, order)
    return FieldOperator(field, dt.operator, dt.expression)
