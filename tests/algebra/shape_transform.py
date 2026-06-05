from algebra.space import Space, ShapeTransform


def test_none_transform_is_identity():
    space = Space((4, 5))
    shape = (3, 4, 5)
    assert ShapeTransform.NONE.transform(space, shape) == shape
    assert ShapeTransform.NONE.reverse(space, shape) == shape


def test_increase_rank_inserts_component_axis_before_space():
    space = Space((4, 5))  # ndim = 2
    # scalar field
    assert ShapeTransform.INCREASE_RANK.transform(space, (4, 5)) == (2, 4, 5)
    # vector field
    assert ShapeTransform.INCREASE_RANK.transform(space, (3, 4, 5)) == (
        3,
        2,
        4,
        5,
    )


def test_reduce_rank_removes_component_axis_before_space():
    space = Space((4, 5))  # ndim = 2
    assert ShapeTransform.REDUCE_RANK.transform(space, (2, 4, 5)) == (4, 5)
    assert ShapeTransform.REDUCE_RANK.transform(space, (3, 2, 4, 5)) == (
        3,
        4,
        5,
    )


def test_reverse_is_inverse_of_transform_increase():
    space = Space((4, 5))
    for shape in [(4, 5), (3, 4, 5), (7, 3, 4, 5)]:
        out = ShapeTransform.INCREASE_RANK.transform(space, shape)
        assert ShapeTransform.INCREASE_RANK.reverse(space, out) == shape


def test_reverse_is_inverse_of_transform_reduce():
    space = Space((4, 5))
    for shape in [(2, 4, 5), (3, 2, 4, 5)]:
        out = ShapeTransform.REDUCE_RANK.transform(space, shape)
        assert ShapeTransform.REDUCE_RANK.reverse(space, out) == shape


def test_transform_works_for_1d_space():
    space = Space((6,))  # ndim = 1
    assert ShapeTransform.INCREASE_RANK.transform(space, (6,)) == (1, 6)
    assert ShapeTransform.REDUCE_RANK.transform(space, (1, 6)) == (6,)
