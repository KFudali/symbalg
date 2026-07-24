import numpy as np

from algebra.domain import bcs
from algebra.systems.systems import LinearSystem
from discrete.fd.domain.fd_domain import FDDomain
from discrete.fd.domain.bcs import FDBCTool
from discrete.fd.operator import dx
from tools.geometry import StructuredGridND


def _make_setup(field_shape: tuple[int, ...]):
    grid = StructuredGridND((10, 10), (0.1, 0.1))
    domain = FDDomain(grid)
    bc_tool = FDBCTool(domain.space, domain.boundaries)
    lap = dx.laplace(domain.space, order=2, h=0.1)
    rhs = np.zeros(field_shape, dtype=float)
    system = LinearSystem(lap, rhs)
    return domain, bc_tool, system


def test_post_solve_dirichlet_scalar():
    domain, bc_tool, _ = _make_setup((10, 10))
    left_id, _ = domain.ax_boundaries(ax=0)
    bc = bcs.BoundaryCondition(3.0, bcs.BCType.DIRICHLET, left_id)

    field = np.zeros((10, 10))
    bc_tool.normalize([bc], field)

    expected = np.zeros((10, 10))
    expected[0, :] = 3.0
    assert np.array_equal(field, expected)


def test_post_solve_dirichlet_vector_field():
    """Each component of a vector field must have the boundary along the
    space axis set to the BC value, not along the component axis.
    """
    domain, bc_tool, _ = _make_setup((10, 10))
    left_id, _ = domain.ax_boundaries(ax=0)
    bc = bcs.BoundaryCondition(7.0, bcs.BCType.DIRICHLET, left_id)

    field = np.zeros((2, 10, 10))
    bc_tool.normalize([bc], field)

    expected = np.zeros((2, 10, 10))
    expected[:, 0, :] = 7.0
    assert np.array_equal(field, expected)


def test_post_solve_dirichlet_tensor_field():
    domain, bc_tool, _ = _make_setup((10, 10))
    _, right_id = domain.ax_boundaries(ax=1)
    bc = bcs.BoundaryCondition(2.0, bcs.BCType.DIRICHLET, right_id)

    field = np.zeros((3, 2, 10, 10))
    bc_tool.normalize([bc], field)

    expected = np.zeros((3, 2, 10, 10))
    expected[:, :, :, -1] = 2.0
    assert np.array_equal(field, expected)


def test_apply_dirichlet_vector_rhs_zeroes_space_boundary_per_component():
    """`dirichlet.apply` zeroes the boundary slice of rhs as part of the
    contribution. For a vector rhs, the zeroed slice must lie along the
    space axis of the BC, not along the component axis.
    """
    domain, bc_tool, system = _make_setup((2, 10, 10))
    left_id, _ = domain.ax_boundaries(ax=0)
    bc = bcs.BoundaryCondition(1.0, bcs.BCType.DIRICHLET, left_id)

    # Pre-populate rhs with a known non-zero pattern so we can inspect what
    # was overwritten.
    system.rhs[...] = 5.0

    lhs = bc_tool.apply_bcs([bc], system.lhs, system.rhs)
    new_system = LinearSystem(lhs, system.rhs)

    # Per-component, the first row along space axis 0 must have been
    # touched by `_add_boundary_rhs_contribution` (set to zero).
    for comp in range(2):
        assert np.all(new_system.rhs[comp, 0, :] == 0.0)

    # Interior rows (away from any stencil reach) must remain at 5.0.
    assert np.all(new_system.rhs[:, 5, :] == 5.0)


def test_apply_dirichlet_modifies_lhs_stencil_consistently():
    """The lhs stencil rewrite is rhs-independent. The result for a vector
    rhs must equal the result for a scalar rhs.
    """
    domain, bc_tool, scalar_system = _make_setup((10, 10))
    _, right_id = domain.ax_boundaries(ax=1)
    bc = bcs.BoundaryCondition(4.0, bcs.BCType.DIRICHLET, right_id)

    lhs = bc_tool.apply_bcs([bc], scalar_system.lhs, scalar_system.rhs)
    scalar_out = LinearSystem(lhs, scalar_system.rhs)

    _, _, vector_system = _make_setup((2, 10, 10))
    lhs = bc_tool.apply_bcs([bc], vector_system.lhs, vector_system.rhs)
    vector_out = LinearSystem(lhs, scalar_system.rhs)

    for ax in range(2):
        assert scalar_out.lhs.stencils[ax].lefts == vector_out.lhs.stencils[ax].lefts
        assert scalar_out.lhs.stencils[ax].rights == vector_out.lhs.stencils[ax].rights
        assert (
            scalar_out.lhs.stencils[ax].interior == vector_out.lhs.stencils[ax].interior
        )


def test_apply_neumann_vector_rhs():
    """Neumann BC adds a per-component contribution to the rhs at the
    space-boundary slice. Must not modify the component axis.
    """
    domain, bc_tool, system = _make_setup((2, 10, 10))
    left_id, _ = domain.ax_boundaries(ax=0)
    bc = bcs.BoundaryCondition(1.0, bcs.BCType.NEUMANN, left_id)

    lhs = bc_tool.apply_bcs([bc], system.lhs, system.rhs)
    new_system = LinearSystem(lhs, system.rhs)

    # First space row of each component should be modified (non-zero).
    for comp in range(2):
        assert np.any(new_system.rhs[comp, 0, :] != 0.0)

    # Interior should remain zero.
    assert np.all(new_system.rhs[:, 5, :] == 0.0)
