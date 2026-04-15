from algebra.field import Field
from algebra.symbolic import SymbolicExpression, SymbolicOperator, AffineOperator
from algebra.expression import ScalarExpression
from ..dx import eye


def euler(field: Field, order: int = 1) -> AffineOperator:
    if order == 1:
        dt = SymbolicExpression(ScalarExpression(field.space.time.dt))
        mass = SymbolicOperator(eye(field.space, field.components)) / dt
        const = field.past(1).value() / dt
        return AffineOperator(mass, const)
    return None
