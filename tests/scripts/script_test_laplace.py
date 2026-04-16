from discrete import fd
from fieldspace import FieldSpace, dx, systems, monitors
from tools.geometry import StructuredGridND
from tools.time.series import ConstDtTimeSeries

grid = StructuredGridND((10, 10), (0.1, 0.1))
time = ConstDtTimeSeries(dt = 0.01)
space = fd.FdDiscreteSpace(grid, time)
top, bottom = space.domain.ax_boundary(ax = 0)
left, right = space.domain.ax_boundary(ax = 1)

fieldspace = FieldSpace(space)
F = fieldspace.field(components = 1)

rhs = fieldspace.field(components = 1)
lhs = dx.laplace(F)

equation = systems.les(lhs, rhs)

top_bc = systems.bcs.dirichlet(top, 10)
bot_bc = systems.bcs.dirichlet(bottom, 0)
left_bc = systems.bcs.neumann(left, -20)
right_bc = systems.bcs.neumann(right, 20)
bcs = [top_bc, bot_bc, left_bc, right_bc]

equation.add_bcs(bcs)

F.set_value(equation.solve()).perform()

monitors.plot_field(F)
