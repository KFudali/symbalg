# AGENTS.md

Agent guide for `symbalg`. Every line is hard-earned — if you're unsure, trust it over guessing.

## Environment

- **Virtualenv:** `~/.virtualenvs/algfields3.12/` — activate first.
- **Tests:** `pytest` (configured by `pytest.ini`, which sets `pythonpath = src`). All 325 tests pass.
- **Linter:** `pylint` (v4.0.5)
- **Type checker:** `ty` (v0.0.34)
- **Formatter:** `black` (v26.3.1)
- Run order: `pylint src/ && black --check src/ && ty src/ && pytest`.

## Project purpose

Abstract algebra over `np.ndarray`. Users write numerical methods against symbolic `Operator`/`Expression`/`Field` objects. The `Discretization` supplies concrete implementations (`laplace`, `grad`, `div`, time derivatives) at `eval()`/`apply()` time. User code never touches arrays directly.

## Architecture rules

- `algebra` MUST NOT import from `discrete`.
- `discrete` may depend on `algebra` and `tools` only.
- `fieldspace` is the only layer that wires `discrete` + `algebra` for users.
- `tools` stays dependency-free of `algebra`, `discrete`, and `fieldspace`.
- State mutations are returned as `LazyAction`; the caller is responsible for `.perform()`.

## Package layout (`src/`)

- `algebra/` — Core abstract algebra. Backend-agnostic.
  - `algebra/space/` — `Space`, `FieldShape`, `FieldShaped`, `ShapeTransform`
  - `algebra/field.py` — `Field` (wraps `ValueBuffer`), `AbstractField`
  - `algebra/expression/core/expression.py` — `Expression` (ABC), `ConstExpression`, `CallableExpression`
  - `algebra/expression/symbolic/` — `SymbolicExpression`, custom AST nodes
  - `algebra/operator/core/operator.py` — `Operator` (ABC), `TOperator` type alias
  - `algebra/operator/symbolic.py` — `SymbolicOperator`
  - `algebra/operator/affine_operator.py` — `AffineOperator`
  - `algebra/domain/` — `Domain` (ABC), `BoundaryTool` (ABC), `BoundaryCondition`, `BCType`, boundary types, `DomainOperator`
  - `algebra/systems/` — `LinearEquation`, `LinearSystem`, `CGSolver`, `SystemConstraint`
  - `algebra/exceptions.py` — `ShapeMismatchError`
- `discrete/core/` — Abstract bases: `Discretization`, `DxOperators`, `DtOperators`, `DiscreteTime`
- `discrete/fd/` — Finite-difference implementations: `FdDiscretization`, `StencilOperator`, `FDDomain`, stencils, FD BCs
  - `discrete/fd/stencil/stencil_operator.py` — `StencilOperator` (the concrete FD operator)
  - `discrete/fd/domain/bcs/` — `FDBCTool`, dirichlet/neumann implementations
- `fieldspace/` — User-facing facade: `FieldSpace` wires a `Discretization` to factories (`fields`, `dx`, `dt`, `systems`, `time`, `monitors`).
  - `fieldspace/systems.py` — `SystemFactory.les(...)` returns `LinearEquation`
- `tools/` — Dependency-free utilities: generic symbolic AST (`Symbolic`, `ValueNode`, etc.), `ValueBuffer`/`DequeValueBuffer`, `StructuredGridND`, region helpers, `AdvanceableSeries`, `LazyAction`.

## Key entry points

| Class / symbol            | Location                                              |
|---------------------------|-------------------------------------------------------|
| `Field`                   | `algebra/field.py:32`                                 |
| `Space`                   | `algebra/space/space.py:6`                            |
| `FieldShape`              | `algebra/space/fieldshaped.py:6`                      |
| `ShapeTransform`          | `algebra/space/shape_transfrom.py:5`                  |
| `Expression` (ABC)        | `algebra/expression/core/expression.py:7`             |
| `Operator` (ABC)          | `algebra/operator/core/operator.py:12`                |
| `SymbolicExpression`      | `algebra/expression/symbolic/symbolic_expression.py:15` |
| `SymbolicOperator`        | `algebra/operator/symbolic.py:16`                     |
| `AffineOperator`          | `algebra/operator/affine_operator.py:12`              |
| `Discretization` (ABC)    | `discrete/core/discretization.py:13`                  |
| `DxOperators` (ABC)       | `discrete/core/dx_operators.py:7`                     |
| `DtOperators` (ABC)       | `discrete/core/dt_operators.py:6`                     |
| `FdDiscretization`        | `discrete/fd/fd_discretization.py:10`                 |
| `StencilOperator`         | `discrete/fd/stencil/stencil_operator.py:13`          |
| `BoundaryCondition`/`BCType` | `algebra/domain/bcs/bcs.py:17` / `.py:11`          |
| `BoundaryTool` (ABC)      | `algebra/domain/bcs/boundary_tool.py:10`              |
| `FDBCTool`                | `discrete/fd/domain/bcs/bc_tool.py:23`                |
| `FieldSpace`              | `fieldspace/fieldspace.py:9`                          |
| `LinearEquation`          | `algebra/systems/equation.py:14`                      |
| `CGSolver`                | `algebra/systems/solvers.py:24`                       |
| `ShapeMismatchError`      | `algebra/exceptions.py:1`                             |

## Typical user flow

Reference: `tests/scripts/script_test_diffusion.py`.

1. `grid = StructuredGridND(...)`; `discrete = fd.FdDiscretization(grid)`
2. `s = FieldSpace(discrete)` — exposes `s.fields`, `s.dx`, `s.dt`, `s.systems`, `s.time`, `s.monitors`
3. `F = s.fields.scalar()` — returns `Field` backed by `DequeValueBuffer`
4. Compose operators symbolically: `lap = s.dx.laplace()`, `dt_op = s.dt.explicit(F, order=2)`, `lhs = dt_op - L * lap`
5. Form equation: `eq = s.systems.les(lhs, rhs.value(), bcs)`. BCs via `s.systems.bc.dirichlet(...)` / `neumann(...)`. `les()` strips affine bias from `lhs` into `rhs`.
6. Solve: `solution = eq.solve(CGSolver())` returns a `SymbolicExpression`. Call `.eval()` to run CG.
7. `F.set_value(solution).perform()` — `LazyAction` writes result into the field buffer.
8. Time loop: `for step in s.time.run(duration=1.0, init_dt=0.01): ...`

## Conventions

- Classes: `PascalCase`; modules: `snake_case`. FD classes prefixed `FD` (e.g., `FdDiscretization`, `FDDomain`, `FDBCTool`).
- Abstracts in `core/` subpackages; concretes in sibling packages (e.g., `discrete/core/` vs `discrete/fd/`).
- Frozen dataclasses for value objects (`Space`, `FieldShape`, `BoundaryCondition`, symbolic nodes).
- Only custom exception: `ShapeMismatchError`; other validation uses `assert`/`ValueError`.
- Operator/Expression algebra built on `Symbolic[T]` using frozen-dataclass node trees (`ValueNode`/`UnaryNode`/`BinaryNode`). Lazy `.resolve()` / `.eval()`. `_compatible()` gates combinations.

## Tests

325 tests across `tests/`. Layout mirrors `src/`.

- `tests/conftest.py` — `MockOperator` test double (not in `tests/algebra/`).
- `tests/algebra/field/` — `test_field.py`
- `tests/algebra/space/` — shape transform, shape utils
- `tests/algebra/symbolic/expression/` — symbolic expression algebra, magics, unary
- `tests/algebra/symbolic/operator/` — `test_symbolic_operator.py`, `test_affine_operator.py`
- `tests/discrete/fd/stencil/` — stencil, ax_stencil, laplace
- `tests/discrete/fd/stencil/operators/` — grad, div, ders, combine
- `tests/discrete/fd/operator/` — `test_as_array.py`
- `tests/discrete/fd/operator/bcs/` — `test_bc_tool.py`
- `tests/discrete/fd/operator/dt/` — `test_dt_explicit.py`
- `tests/tools/` — `test_symbolic.py`
- `tests/tools/buffer/` — stacked proxy, component proxy buffer tests
- `tests/tools/region/` — `test_region.py`, `test_region_utils.py`
- `tests/scripts/` — runnable demos (not collected): `script_test_diffusion.py`, `script_test_laplace.py`, `script_test_poisson.py`, `script_test_navier_stokes.py`, `script_test_navier_stokes_conv.py`, `script_test_neumann_only.py`. Used as end-to-end usage examples.

Pytest collects `test_*.py`; demo scripts use `script_test_*.py` prefix to opt out.
