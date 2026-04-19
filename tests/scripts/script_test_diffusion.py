from discrete import fd
from fieldspace import FieldSpace, dx, dt, systems, monitors
from tools.geometry import StructuredGridND

N = 20
grid = StructuredGridND((N, N), (0.1, 0.1))
space = fd.FdDiscreteSpace(grid)
top, bottom = space.domain.ax_boundaries(ax = 0)
left, right = space.domain.ax_boundaries(ax = 1)

fieldspace = FieldSpace(space)

F = fieldspace.field(components = 1)

L = 1.0
f_dx = dx.laplace(F)
f_dt = dt.euler(F)

lhs = f_dt - L * f_dx
rhs = fieldspace.field(components = 1, init_value=0.0)

equation = systems.les(lhs, rhs.value())

top_bc = systems.bcs.dirichlet(top, 10)
bot_bc = systems.bcs.dirichlet(bottom, 0)
left_bc = systems.bcs.neumann(left, -20)
right_bc = systems.bcs.neumann(right, 20)
bcs = [top_bc, bot_bc, left_bc, right_bc]

equation.add_bcs(bcs)

f_history = monitors.FieldMonitor2D(F)
for step in fieldspace.time.run(duration = 1.0, init_dt = 0.01):
    solution = equation.solve()
    F.set_value(solution).perform()
f_history.animate()
