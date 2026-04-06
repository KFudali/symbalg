from algebra.space import Space, SpaceObject
from algebra.expression.expression import Expression


class SpaceExpression(SpaceObject, Expression):
    def __init__(self, space: Space, output_components: int):
        SpaceObject.__init__(self, space)
        output_shape = (output_components, *space.shape)
        Expression.__init__(self, output_shape)
