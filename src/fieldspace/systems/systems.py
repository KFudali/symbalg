from discrete.core.bcs import BoundaryCondition, BCTool
from algebra.expression import Expression
from algebra.operator import Operator
from .les import LES


class SystemFactory:
    def __init__(self, bc_tool: BCTool):
        self._bc_tool = bc_tool

    def les(self, lhs: Operator, rhs: Expression, bcs: list[BoundaryCondition]) -> LES:
        return LES(self._bc_tool, lhs, rhs, bcs)
