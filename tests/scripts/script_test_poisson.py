from fieldspace import FieldSpace
from tools.geometry import StructuredGridND
from discrete import fd

N = 100
grid = StructuredGridND((N, N), (0.1, 0.1))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)

top, bottom = discrete.domain.ax_boundaries(ax=0)
left, right = discrete.domain.ax_boundaries(ax=1)
top_bc = s.systems.bc.dirichlet(top, 0.0)
bot_bc = s.systems.bc.dirichlet(bottom, 0.0)
left_bc = s.systems.bc.dirichlet(left, 0.0)
right_bc = s.systems.bc.dirichlet(right, 0.0)
bcs = [top_bc, bot_bc, left_bc, right_bc]


F = s.fields.scalar()
L = 1.0
lap = s.dx.laplace()

lhs = L * lap
rhs = s.fields.scalar(init_value=1.0)

equation = s.systems.les(lhs, rhs.value(), bcs)

solution = equation.solve()
F.set_value(solution).perform()

s.monitors.plot_field(F)
