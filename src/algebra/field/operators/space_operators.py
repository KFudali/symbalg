from algebra.core.space import Space, FieldShape
from algebra.exceptions import ShapeMismatchError
from .operator import Operator


class SpaceOperator(Operator):
    def __init__(self, input_shape: FieldShape, output_shape: FieldShape):
        Operator.__init__(self, input_shape.shape, output_shape.shape)
        if input_shape.space != output_shape.space:
            raise ShapeMismatchError("Input and output space shapes have to match.")
        self._input_fieldshape = input_shape
        self._output_fieldshape = output_shape

    @property
    def space(self) -> Space:
        return self._input_fieldshape.space

    @property
    def input_rank(self) -> int:
        return self._input_fieldshape.rank

    @property
    def output_rank(self) -> int:
        return self._output_fieldshape.rank


class LapLikeOperator(SpaceOperator):
    def __init__(self, input_shape: FieldShape):
        super().__init__(input_shape, input_shape)


class GradLikeOperator(SpaceOperator):
    def __init__(self, input_shape: FieldShape):
        # Gradient raises the tensor rank by 1: append a new rank axis of
        # length space.ndim after the existing rank axes.
        space = input_shape.space
        output_shape = FieldShape(
            space=space,
            rank_shapes=(*input_shape.rank_shapes, space.ndim),
        )
        super().__init__(input_shape, output_shape)


class DivLikeOperator(SpaceOperator):
    def __init__(self, input_shape: FieldShape):
        if input_shape.rank < 1:
            raise ShapeMismatchError(
                (
                    "To create DivLikeOperator input field must have rank >= 1 "
                    "(at least one tensor axis to contract).",
                    f"input rank_shapes: {input_shape.rank_shapes}",
                )
            )
        space = input_shape.space
        contracted_dim = input_shape.rank_shapes[-1]
        if contracted_dim != space.ndim:
            raise ShapeMismatchError(
                (
                    "To create DivLikeOperator the last rank axis (the one to "
                    f"be contracted) must have length equal to space.ndim. "
                    f"space ndim: {space.ndim}, last rank dim: {contracted_dim}",
                    f"input rank_shapes: {input_shape.rank_shapes}",
                )
            )
        # Divergence contracts the last rank axis (the one just before the
        # spatial axes) with the spatial gradient, lowering rank by 1.
        output_shape = FieldShape(
            space=space,
            rank_shapes=input_shape.rank_shapes[:-1],
        )
        super().__init__(input_shape, output_shape)


TSpaceOperator = TypeVar("TSpaceOperator", bound=SpaceOperator)
