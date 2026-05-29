from discrete import fd
from fieldspace import FieldSpace, dx, dt, systems, monitors
from tools.geometry import StructuredGridND

grid = StructuredGridND((10, 10), (0.1, 0.1))
space = fd.FdDiscreteSpace(grid)
top, bottom = space.domain.ax_boundaries(ax=0)
left, right = space.domain.ax_boundaries(ax=1)

fieldspace = FieldSpace(space)

top_bc = systems.bcs.dirichlet(top, 10)
bot_bc = systems.bcs.dirichlet(bottom, 0)
left_bc = systems.bcs.dirichlet(left, 0)
right_bc = systems.bcs.dirichlet(right, 0)
u_bcs = [top_bc, bot_bc, left_bc, right_bc]

top_bc = systems.bcs.neumann(top, 0)
bot_bc = systems.bcs.neumann(bottom, 0)
left_bc = systems.bcs.neumann(left, 0)
right_bc = systems.bcs.neumann(right, 0)
fi_bcs = [top_bc, bot_bc, left_bc, right_bc]

u = fieldspace.field(components=3, init_value=0.0)
f = fieldspace.field(components=3, init_value=0.0)
p = fieldspace.field(components=1, init_value=0.0)
fi = fieldspace.field(components=1, init_value=0.0)


t_val = fieldspace.time.dt()
# Step 1
NU = 0.01
dudt = dt.bfd(u, order=2)
step_1 = systems.les(lhs=dudt - NU * dx.laplace(u), rhs=-dx.grad(p) + f.value())
step_1.add_bcs(u_bcs)
# Step 2
step_2 = systems.les(lhs=dx.laplace(fi), rhs=(3.0 / 2.0 * dt_val) * dx.div(u))
step_2.add_bcs(fi_bcs)
# Step 3
p_update = p.set_value(p.past(1).value() + fi.value() - NU * dx.div(u))

u_history = monitors.FieldMonitor2D(u)
p_history = monitors.FieldMonitor2D(p)

for time in fieldspace.time.run(duration=1.0, init_dt=0.01):
    u.set_value(step_1.solve()).perform()
    fi.set_value(step_2.solve()).perform()
    p_update.perform()
p_history.animate()
