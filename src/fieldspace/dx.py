from algebra.field import FieldOperator
from discrete import DiscreteField, DiscreteSpace

def laplace(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    operator = field.discrete.dx.laplace(field.components)
    return FieldOperator(field, operator)

def grad(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    operator = field.discrete.dx.grad(field.components)
    return FieldOperator(field, operator)

def div(field: DiscreteField) -> FieldOperator[DiscreteSpace]:
    operator = field.discrete.dx.div(field.components)
    return FieldOperator(field, operator)
