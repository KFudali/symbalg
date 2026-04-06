from ..field import Field
from .field_operator_expression import FieldOperatorExpression
import algebra

def laplace(field: Field) -> FieldOperatorExpression:
    laplace = field.space.discretization.operators.laplace()
    field_shaped = algebra.operator.ComponentWiseOperator(laplace, field.components)
    return FieldOperatorExpression(field.value(), field_shaped)
