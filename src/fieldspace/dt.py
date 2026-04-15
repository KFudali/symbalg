from algebra.field import FieldOperator
from discrete import DiscreteField, DiscreteSpace

def euler(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    dt = field.discrete.dt.explicit_euler(field)
    return FieldOperator(field, dt.operator, dt.expression)
