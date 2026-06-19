symbalg is a small library written initially as an exercise in linear algebra and programming. The idea behind the library is to offer abstractions over linear algebra that allow user to assemble his numerical methods without specyfing discretization schemes. The discretization should remain abstract as the numerical method should not depend on how constructs like Laplacian or Gradient are implemented. 

The above target resulted in library that is based on two main concepts:
 - Expression - lazy class that upon calling .eval() returns some field as numpy array.
 - Operator - class that can .apply(input, output) to fill output based on input. Basic example of an Operator is the Laplace operator. 

For both of those core concepts Symbolic versions were implemented to simplify assembly of numerical methods and eliminate need for most for loops.

The core class exposed to the user is the FieldSpace(). it has to be initialzied with a discretization object. So far the only available discretization is simple FiniteDifference but FEM discretizaiton is currently being prototyped.

The FieldSpace class allows to create Field object of any rank. It also serves as a factory for equations and operators (as it calls algebra abstracts from discretization object). Apart from specyfing domain regions and boundaries user should depend only on FieldSpace object to assemble a numerical scheme. FieldSpace also exposes and interface for running time loops with its .time property. 

For convenience FieldSpace also exposes monitors object that allows to plot resulting fields and track field transformation over time.

Example numerical schemes implemetned using symbalg can be found tests/scripts

### Navier-stokes example over a square domain###
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
