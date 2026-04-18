from discrete import fd
from fieldspace import FieldSpace, dx, systems, monitors
from tools.geometry import StructuredGridND
from tools.time.series import ConstDtTimeSeries

grid = StructuredGridND((10, 10), (0.1, 0.1))
time = ConstDtTimeSeries(dt = 0.01)
space = fd.FdDiscreteSpace(grid, time)
top, bottom = space.domain.ax_boundaries(ax = 0)
left, right = space.domain.ax_boundaries(ax = 1)

fieldspace = FieldSpace(space)
F = fieldspace.field(components = 1)

rhs = fieldspace.field(components = 1, init_value=100.0)
lhs = dx.laplace(F)

equation = systems.les(lhs, rhs.value())

top_bc = systems.bcs.dirichlet(top, 0)
bot_bc = systems.bcs.dirichlet(bottom, 0)
left_bc = systems.bcs.dirichlet(left, 0)
right_bc = systems.bcs.dirichlet(right, 0)
bcs = [top_bc, bot_bc, left_bc, right_bc]

equation.add_bcs(bcs)

solution = equation.solve()
F.set_value(solution).perform()
monitors.plot_field_2d(F)
