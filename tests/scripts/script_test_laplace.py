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
sol = solution.eval()


import matplotlib.pyplot as plt
import numpy as np
u = sol
nx, ny = grid.shape
dx, dy = grid.spacing
x = np.linspace(0, (nx-1)*dx, nx)
y = np.linspace(0, (ny-1)*dy, ny)
X, Y = np.meshgrid(x, y)
U = u.reshape(X.shape)
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, U, cmap='viridis', edgecolor='k', linewidth=0.5)
ax1.set_title("Conjugate Gradient")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='u')
plt.tight_layout()
plt.show()
