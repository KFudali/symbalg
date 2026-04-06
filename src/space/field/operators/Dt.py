from ..field import Field
from .field_operator_expression import FieldOperatorExpression

from space.time import explicit


def euler(field: Field) -> FieldOperatorExpression:
    return explicit.EulerTimeDer(field)
