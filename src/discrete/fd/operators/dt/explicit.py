from algebra.expression import Expression
from algebra.field import Field

from algebra.symbolic import SymbolicExpression, SymbolicOperator, AffineOperator
from .. import dx


def euler(field: Field, dt: Expression) -> AffineOperator:
    dt_exp = SymbolicExpression.wrap(dt)
    eye = SymbolicOperator.wrap(dx.eye(field.space))
    value = SymbolicExpression.wrap(field.past(1).value())
    mass = eye / dt_exp
    const = value / dt_exp
    return AffineOperator(mass, const)


def bfd(field: Field, dt: Expression, order: int = 1) -> AffineOperator:
    if order == 1:
        return euler(field, dt)
    if order == 2:
        dt_exp = SymbolicExpression.wrap(dt)
        eye = SymbolicOperator.wrap(dx.eye(field.space))
        mass = eye / dt_exp
        prev_1 = SymbolicExpression.wrap(field.past(1).value())
        prev_2 = SymbolicExpression.wrap(field.past(2).value())
        op = mass * (3.0 / 2.0 * dt_exp)
        exp = (-4.0 * prev_1 + prev_2) / (2.0 * dt_exp)
        return AffineOperator(op, exp)
    return NotImplemented
