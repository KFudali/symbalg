from fieldspace import FieldSpace
from tools.geometry import StructuredGridND
from discrete import fd

N = 20
grid = StructuredGridND((N, N), (0.1, 0.1))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)

top, bottom = discrete.domain.ax_boundaries(ax=0)
left, right = discrete.domain.ax_boundaries(ax=1)
top_bc = s.systems.bc.dirichlet(top, 10.0)
bot_bc = s.systems.bc.dirichlet(bottom, 0.0)
left_bc = s.systems.bc.neumann(left, -20.0)
right_bc = s.systems.bc.neumann(right, 20.0)
bcs = [top_bc, bot_bc, left_bc, right_bc]


F = s.fields.scalar()
L = 1.0
f_dx = s.dx.laplace()
f_dt = s.dt.euler(F)
lhs = f_dt - f_dx
op = lhs.operator.resolve()

rhs = s.fields.scalar()
equation = s.systems.les(lhs, rhs.value(), bcs)

for step in s.time.run(duration=1.0, init_dt=0.01):
    solution = equation.solve()
    F.set_value(solution).perform()
s.monitors.plot_field(F.past(1))