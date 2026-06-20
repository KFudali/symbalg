### About symbalg
symbalg is a small library written initially as an exercise in linear algebra and programming. The idea behind the library is to offer abstractions over linear algebra that allow user to assemble his numerical methods without specyfing discretization schemes. The discretization should remain abstract as the numerical method should not depend on how constructs like Laplacian or Gradient are implemented. 

The above target resulted in library that is based on two main concepts:
 - Expression - lazy class that upon calling .eval() returns some field as numpy array.
 - Operator - class that can .apply(input, output) to fill output based on input. Basic example of an Operator is the Laplace operator. 

For both of those core concepts Symbolic versions were implemented to simplify assembly of numerical methods and eliminate need for most for loops.

### Usage
The core class exposed to the user is the FieldSpace(). it has to be initialzied with a discretization object. So far the only available discretization is simple FiniteDifference but FEM discretizaiton is currently being prototyped.

The FieldSpace class allows to create Field object of any rank. It also serves as a factory for equations and operators (as it calls algebra abstracts from discretization object). Apart from specyfing domain regions and boundaries user should depend only on FieldSpace object to assemble a numerical scheme. FieldSpace also exposes and interface for running time loops with its .time property. 

For convenience FieldSpace also exposes monitors object that allows to plot resulting fields and track field transformation over time.

Example numerical schemes implemetned using symbalg can be found tests/scripts

### Navier-stokes example over a square domain
```python
from fieldspace import FieldSpace
from algebra.systems import solvers, constraints
from tools.geometry import StructuredGridND
from discrete import fd

N = 20
grid = StructuredGridND((N, N), (0.05, 0.05))
discrete = fd.FdDiscretization(grid)
s = FieldSpace(discrete)

cg = solvers.CGSolver()
# Step 1
NU = 0.01
step_1 = s.systems.les(
    lhs=s.dt.explicit(u, order=2) - (NU * s.dx.laplace()),
    rhs=s.dx.grad().of(p) + f.value(),
    bcs=u_bcs,
)
# Step 2
step_2 = s.systems.les(
    lhs=s.dx.laplace(),
    rhs=(3.0 / 2.0 * s.time.dt()) * s.dx.div().of(u),
    bcs=fi_bcs,
    constraints=[fi_cstr],
)
new_p = p.past(1).value() + fi.value() - NU * s.dx.div().of(u)
for time in s.time.run(duration=1.0, init_dt=0.01):
    u.set_value(step_1.solve(cg)).perform()
    fi.set_value(step_2.solve(cg)).perform()
    p.set_value(new_p).perform()
```

### Simple Poisson equation with dirichlet conditions
```python
from fieldspace import FieldSpace
from algebra.systems import solvers
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

solution = equation.solve(solvers.CGSolver())
F.set_value(solution).perform()

s.monitors.plot_field_2d(F)
```

### Future plans
I plan to solve more problems with this library to learn about linear algebra edge cases and whether its concepts can really be programmatically abstracted. With time more script_test files should appear. I am also currently working on FEM discretization 
implementation that will allow usage for complicated gemetries not only FiniteDifference structured grids.