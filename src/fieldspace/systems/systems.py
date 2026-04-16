from algebra import Expression, field
from discrete import DiscreteSpace
from .les import SymbolicLES

def les(
    lhs: field.FieldOperator[DiscreteSpace],
    rhs: Expression
) -> SymbolicLES:
    return SymbolicLES(lhs, rhs)
