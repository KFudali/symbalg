from discrete import fd
from fieldspace import FieldSpace, dx, systems, monitors
from tools.geometry import StructuredGridND

grid = StructuredGridND((10, 10), (0.1, 0.1))
space = fd.FdDiscreteSpace(grid)
top, bottom = space.domain.ax_boundaries(ax = 0)
left, right = space.domain.ax_boundaries(ax = 1)

fieldspace = FieldSpace(space)
F = fieldspace.field(components = 1)

rhs = fieldspace.field(components = 1)
lhs = dx.laplace(F)

equation = systems.les(lhs, rhs.value())

top_bc = systems.bcs.dirichlet(top, 10)
bot_bc = systems.bcs.dirichlet(bottom, 0)
left_bc = systems.bcs.neumann(left, -20)
right_bc = systems.bcs.neumann(right, 20)
bcs = [top_bc, bot_bc, left_bc, right_bc]

equation.add_bcs(bcs)

solution = equation.solve()
F.set_value(solution).perform()
monitors.plot_field_2d(F)
