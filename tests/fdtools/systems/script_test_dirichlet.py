import numpy as np
import matplotlib.pyplot as plt
from systems import bcs, solve_les
import region
import operators

N = 100
fieldshape = (N, N)
h = 0.1
field = np.zeros(dtype=float, shape=fieldshape)
rhs = np.zeros_like(field)
laplace = operators.laplace(order=2, h=h, ndim = 2)

#zero dirichlet boundaries
top = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, h, 0, -1, 10.0)
bottom = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, h, 0, 1, 0.0)
left = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, h, 1, 1, 0.0, True)
right = bcs.BoundaryCondition(bcs.BConType.DIRICHLET, h, 1, -1, 0.0, True)

result = solve_les(laplace, rhs, ((top, bottom), (left, right)))

top = region.boundary(2, 0, -1, True)
bottom = region.boundary(2, 0, 1, True)
left = region.boundary(2, 1, -1)
right = region.boundary(2, 1, 1)
dir_values = {top: 10.0, bottom: 0.0}
neu_values = {left: 10.0, right: 0.0}
for reg, value in dir_values.items():
    assert np.allclose(result[reg], value)

x = np.linspace(0, (N-1)*h, N)
y = np.linspace(0, (N-1)*h, N)
X, Y = np.meshgrid(x, y)
U = result 
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