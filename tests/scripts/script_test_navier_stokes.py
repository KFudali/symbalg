from fieldspace import FieldSpace
from algebra.systems import solvers, constraints
from tools.geometry import StructuredGridND
from discrete import fd

N = 20
grid = StructuredGridND((N, N), (0.05, 0.05))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)
top, bottom = discrete.domain.ax_boundaries(ax=0)
left, right = discrete.domain.ax_boundaries(ax=1)

top_bc = s.systems.bc.dirichlet(top, [10.0, 0.0])
bot_bc = s.systems.bc.dirichlet(bottom, [0.0, 0.0])
left_bc = s.systems.bc.dirichlet(left, [0.0, 0.0])
right_bc = s.systems.bc.dirichlet(right, [0.0, 0.0])
u_bcs = [top_bc, bot_bc, left_bc, right_bc]

top_bc = s.systems.bc.neumann(top, 0)
bot_bc = s.systems.bc.neumann(bottom, 0)
left_bc = s.systems.bc.neumann(left, 0)
right_bc = s.systems.bc.neumann(right, 0)
fi_bcs = [top_bc, bot_bc, left_bc, right_bc]
fi_cstr = constraints.FixedMeanConstraint()

u = s.fields.vector(init_value=0.0)
f = s.fields.vector(init_value=0.0)
p = s.fields.scalar(init_value=0.0)
fi = s.fields.scalar(init_value=0.0)

cg = solvers.CGSolver()
# Step 1
NU = 0.01
step_1 = s.systems.les(
    lhs=s.dt.explicit(u, order=2) - (NU * s.dx.laplace()),
    rhs= -s.dx.grad().of(p) + f.value(),
    bcs=u_bcs,
)
# Step 2
step_2 = s.systems.les(
    lhs=s.dx.laplace(),
    rhs=(3.0 / (2.0 * s.time.dt())) * s.dx.div().of(u),
    bcs=fi_bcs,
    constraints=[fi_cstr],
)

new_p = p.past(1).value() + fi.value() - NU * s.dx.div().of(u)
for time in s.time.run(duration=1.0, init_dt=0.01):
    u.set_value(step_1.solve(cg)).perform()
    fi.set_value(step_2.solve(cg)).perform()
    p.set_value(new_p).perform()
s.monitors.plot_field_2d(u, "u")
s.monitors.plot_field_2d(p, "p")
s.monitors.show()
