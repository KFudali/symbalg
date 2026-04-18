from algebra.field import FieldOperator
from discrete import DiscreteField, DiscreteSpace

def euler(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    dt = field.space.dt.explicit_euler(field, field.space.time, order=1)
    return FieldOperator(field, dt.operator, dt.expression)
