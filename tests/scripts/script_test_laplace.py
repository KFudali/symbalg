from discrete import fd
from fieldspace import FieldSpace, dx
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

equation = systems.LES(lhs, rhs)

top_bc = systems.bcs.BC(top, systems.bcs.BCType.DIRICHLET, value = 10)
bot_bc = systems.bcs.BC(bottom, systems.bcs.BCType.DIRICHLET, value = 0)
left_bc = systems.bcs.BC(left, systems.bcs.BCType.NEUMANN, value = 20)
right_bc = systems.bcs.BC(right, systems.bcs.BCType.NEUMANN, value = -20)
bcs = [top_bc, bot_bc, left_bc, right_bc]
equation.apply_bcs(bcs)

F.set_value(equation.solve()).perform()
monitors.plotter.plot_field(F)
