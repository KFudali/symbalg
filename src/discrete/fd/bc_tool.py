from discrete.core import BCTool
from algebra.space import Space, SpaceObject
from algebra.systems.bcs import BoundaryCondition
from .domain import FDDomain
from .space_stencil_operator import SpaceStencilOperator

class FDBCTool(BCTool, SpaceObject[FDDomain]):
    def __init__(self, space: Space[FDDomain]):
        super().__init__(space)

    def apply(self, bc: BoundaryCondition, operator: SpaceStencilOperator):
        pass
