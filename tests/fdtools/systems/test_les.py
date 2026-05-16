import numpy as np

from systems import bcs, solve_les
import region
import operators


def test_les_with_dirichlet():

    fieldshape = (10, 10)
    h = 0.01
    field = np.zeros(dtype=float, shape=fieldshape)
    rhs = np.zeros_like(field)
    laplace = operators.laplace(order=2, h=h, ndim=2)

    # zero dirichlet boundaries
    top = bcs.BoundaryCondition(
        bcs.BConType.DIRICHLET, bcs.Boundary(0, -1, h, include_corners=False), 10.0
    )
    bottom = bcs.BoundaryCondition(
        bcs.BConType.DIRICHLET, bcs.Boundary(0, 1, h, include_corners=False), 0.0
    )
    left = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, bcs.Boundary(1, 1, h), 0.0)
    right = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, bcs.Boundary(1, -1, h), 0.0)

    result = solve_les(laplace, rhs, ((top, bottom), (left, right)))

    top = region.boundary(2, 0, -1, True)
    bottom = region.boundary(2, 0, 1, True)
    left = region.boundary(2, 1, -1)
    right = region.boundary(2, 1, 1)
    b_values = {top: 10.0, bottom: 0.0, left: 0.0, right: 0.0}

    for reg, value in b_values.items():
        assert np.allclose(result[reg], value)

    interior = region.interior(2, ((1, 1), (1, 1)))
    assert np.allclose(laplace.apply(result)[interior], 0.0, atol=1e-3)


def test_les_with_mixed():
    fieldshape = (10, 10)
    h = 0.01
    field = np.zeros(dtype=float, shape=fieldshape)
    rhs = np.zeros_like(field)
    laplace = operators.laplace(order=2, h=h, ndim=2)

    # zero dirichlet boundaries
    top = bcs.BoundaryCondition(
        bcs.BConType.DIRICHLET, bcs.Boundary(0, -1, h), 10.0
    )
    bottom = bcs.BoundaryCondition(
        bcs.BConType.DIRICHLET, bcs.Boundary(0, 1, h), 0.0
    )
    left = bcs.BoundaryCondition(
        bcs.BConType.NEUMANN,
        bcs.Boundary(1, 1, h, include_corners=False),
        -20.0,
    )
    right = bcs.BoundaryCondition(
        bcs.BConType.NEUMANN,
        bcs.Boundary(1, -1, h, include_corners=False),
        20.0,
    )

    result = solve_les(laplace, rhs, ((top, bottom), (left, right)))

    top = region.boundary(2, 0, -1, True)
    bottom = region.boundary(2, 0, 1, True)
    left = region.boundary(2, 1, -1)
    right = region.boundary(2, 1, 1)
    dir_values = {top: 10.0, bottom: 0.0}
    neu_values = {left: 10.0, right: 0.0}
    for reg, value in dir_values.items():
        assert np.allclose(result[reg], value)

    interior = region.interior(2, ((1, 1), (1, 1)))
    assert np.allclose(laplace.apply(result)[interior], 0.0, atol=1e-3)
