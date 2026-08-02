from fieldspace import FieldSpace
from algebra.systems import solvers, constraints
from tools.geometry import StructuredGridND
from discrete import fd
from algebra.field import to_operator, ArrayOperator
from algebra.space import ShapeTransform

N = 20
grid = StructuredGridND((N, N), (0.05, 0.05))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)
top, bottom = discrete.domain.ax_boundaries(ax=0)
left, right = discrete.domain.ax_boundaries(ax=1)

top_bc = s.systems.bc.dirichlet(top, [1.0, 0.0])
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
p_hat = s.fields.scalar(init_value=0.0)
p_star = s.fields.scalar(init_value=0.0)
fi = s.fields.scalar(init_value=0.0)
u_grad = s.fields.tensor()
u_grad_trace = s.fields.scalar()

u_grad_update = u_grad.set_value(s.dx.grad().of(u.past(1)))
u_grad_trace_update = u_grad_trace.set_value(u_grad.value().trace())
term_1_op = ArrayOperator(
    u.space,
    ShapeTransform.NONE,
    s.dx.grad().as_array().mat.dot(to_operator(u.past(1)).mat),
)
term_2_op = to_operator(u_grad_trace)

cg = solvers.CGSolver()
NU = 0.01

dt = s.dt.explicit(u, order=2)
dt_lhs = dt.operator.as_array()
dt_rhs = dt.expression

step_1 = s.systems.les(
    lhs=dt_lhs - (NU * s.dx.laplace()).as_array() + term_1_op + term_2_op,
    rhs=-s.dx.grad().of(p_hat) - dt_rhs,
    bcs=u_bcs,
)
step_2 = s.systems.les(
    lhs=s.dx.laplace(),
    rhs=(3.0 / (2.0 * s.time.dt())) * s.dx.div().of(u),
    bcs=fi_bcs,
    constraints=[fi_cstr],
)


for time in s.time.run(duration=1.0, init_dt=0.01):
    p_star.set_value(p.past(1).value()).perform()
    p_hat.set_value(
        p_star.value()
        + ((4.0 / 3.0) * fi.past(1).value())
        - ((1.0 / 3.0) * fi.past(2).value())
    ).perform()
    u.set_value(step_1.solve(cg)).perform()
    u_grad_update.perform()
    u_grad_trace_update.perform()
    fi.set_value(step_2.solve(cg)).perform()
    p.set_value(p_star.value() + fi.value() - (NU * s.dx.div().of(u))).perform()

s.monitors.plot_field_2d(p, "p")
s.monitors.plot_field_2d(u, "u")
s.monitors.show()
