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
bot_bc = s.systems.bc.dirichlet(top, 0.0)
left_bc = s.systems.bc.dirichlet(left, 0.0)
right_bc = s.systems.bc.dirichlet(right, 0.0)
bcs = [top_bc, bot_bc, left_bc, right_bc]

top_bc = s.systems.bc.neumann(top, 0)
bot_bc = s.systems.bc.neumann(bottom, 0)
left_bc = s.systems.bc.neumann(left, 0)
right_bc = s.systems.bc.neumann(right, 0)
fi_bcs = [top_bc, bot_bc, left_bc, right_bc]

u = s.fields.vector(init_value=0.0)
f = s.fields.vector(init_value=0.0)
p = s.fields.scalar(init_value=0.0)
fi = s.fields.scalar(init_value=0.0)
dt_val = s.time.dt()
# Step 1
NU = 0.01
dudt = s.dt.euler(u)
# step_1 = systems.les(lhs=dudt - NU * dx.laplace(u), rhs=-dx.grad(p) + f.value())
# step_1.add_bcs(u_bcs)
# # Step 2
# step_2 = systems.les(lhs=dx.laplace(fi), rhs=(3.0 / 2.0 * dt_val) * dx.div(u))
# step_2.add_bcs(fi_bcs)
# # Step 3
# p_update = p.set_value(p.past(1).value() + fi.value() - NU * dx.div(u))
#
# u_history = monitors.FieldMonitor2D(u)
# p_history = monitors.FieldMonitor2D(p)
#
# for time in fieldspace.time.run(duration=1.0, init_dt=0.01):
#     u.set_value(step_1.solve()).perform()
#     fi.set_value(step_2.solve()).perform()
#     p_update.perform()
# p_history.animate()
