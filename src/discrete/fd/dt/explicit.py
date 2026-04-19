from algebra.field import Field
from algebra.symbolic import SymbolicExpression, SymbolicOperator, AffineOperator
from algebra.core.expression import ScalarExpression
from discrete.core import DiscreteTime
from ..dx import eye


def euler(field: Field, time: DiscreteTime) -> AffineOperator:
    dt = SymbolicExpression(ScalarExpression(time.dt))
    mass = SymbolicOperator(eye(field.space, field.components)) / dt
    const = -field.past(1).value() / dt
    return AffineOperator(mass, const)

def bfd(field: Field, time: DiscreteTime, order: int = 1) -> AffineOperator:
    if order == 1:
        return euler(field, time)
    if order == 2:
        dt = SymbolicExpression(ScalarExpression(time.dt))
        mass = SymbolicOperator(eye(field.space, field.components))
        prev_1 = field.past(1).value()
        prev_2 = field.past(2).value()
        return (3 * mass - 4 * prev_1 + prev_2) / (2 * dt)
    return NotImplemented
