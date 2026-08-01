from algebra.operator import ArrayOperator
from discrete.fd.stencil import StencilOperator
from algebra.space import ShapeTransform


def as_array(operator: StencilOperator) -> ArrayOperator:
    if operator.shape_transform == ShapeTransform.NONE:
        pass
        # sum weights of AxStencisl from all ax into one sparse array. create ArrayOperator with
        # sparse.COO with no components.
    if operator.shape_transform in (
        ShapeTransform.INCREASE_RANK,
        ShapeTransform.REDUCE_RANK,
    ):
        pass
        # transfrom each AxStencil from each axis into spasre array of shape (ndim, n, n)
        # so that ArrayOperator.apply uses mat[ax] on each axis correctly
