import numpy as np
from fieldspace import FieldSpace
from algebra.systems import solvers, constraints
from tools.geometry import StructuredGridND
from discrete import fd

N = 10
grid = StructuredGridND((N, N), (0.1, 0.1))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)

top, bottom = discrete.domain.ax_boundaries(ax=0)
left, right = discrete.domain.ax_boundaries(ax=1)
top_bc = s.systems.bc.neumann(top, 0.0)
bot_bc = s.systems.bc.neumann(bottom, 0.0)
left_bc = s.systems.bc.neumann(left, 0.0)
right_bc = s.systems.bc.neumann(right, 0.0)
bcs = [top_bc, bot_bc, left_bc, right_bc]

F = s.fields.scalar()
L = 1.0
lap = s.dx.laplace()

lhs = L * lap
rhs = s.fields.scalar()
x, y = grid.points()
u = np.cos(x) ** 2 + np.sin(y) ** 2
rhs._value_buffer.set(u)
fix_mean = constraints.FixedMeanConstraint()
equation = s.systems.les(lhs, rhs.value(), bcs, constraints=[fix_mean])

solution = equation.solve(solvers.CGSolver())
F.set_value(solution).perform()

s.monitors.plot_field_2d(F)
s.monitors.show()
