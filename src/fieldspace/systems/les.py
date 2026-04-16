from algebra.systems import LES, bcs
from algebra.field import FieldOperator
from algebra.core.expression import Expression, CallableExpression
from algebra.symbolic import SymbolicExpression

from discrete import DiscreteSpace

class SymbolicLES():
    def __init__(
        self,
        lhs: FieldOperator[DiscreteSpace],
        rhs: Expression
    ):
        self._lhs = lhs
        self._rhs = rhs
        self._bcs = set[bcs.BoundaryCondition]()

    def solve(self) -> SymbolicExpression:
        lhs = self._lhs.operator.fold()
        def eval_rhs():
            return self._rhs.eval() - self._lhs.expression.eval()
        rhs = CallableExpression(eval_rhs, self._rhs.output_shape)
        les = LES(lhs, rhs, self._lhs.field.space.bcs)
        les.add_bcs(list(self._bcs))
        return SymbolicExpression(les.solve())

    def add_bcs(self, bc_list: list[bcs.BoundaryCondition]):
        for bc in bc_list:
            self._bcs.add(bc)
