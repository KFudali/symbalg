# AGENTS.md

Agent-facing guide for the `symbalg` repository. This file is the single source of
truth for conventions, environment, and architecture. It supersedes the previous
`context.md`.

## Environment

- **Virtualenv:** `~/.virtualenvs/algfields3.12/` — activate before running anything.
- **Tests:** `pytest` (configured by `pytest.ini`, which sets `pythonpath = src`).
- **Linter:** `pylint`
- **Type checker:** `ty`
- **Formatter:** `black`

Run lint, type-check, format, and tests before declaring a task complete.

## Project purpose

`symbalg` provides a lightweight abstraction layer for algebra on fields. The base
numerical type is `np.ndarray`, but user code never touches arrays directly. It is
meant for building numerical methods from symbolic abstracts — operators,
expressions, and equations — so algorithms can be expressed independently of the
underlying `Discretization`. Discretizations implement internals such as the space
operators (`laplace`, `grad`, `div`) and time derivatives. Use the `Field` class for
field creation instead of raw numpy arrays.

## Package layout (`src/`)

- **`algebra/`** — Core abstract algebra over `np.ndarray`. Defines shape-aware
  fields, expressions, operators, and a symbolic AST built on top of them.
  Backend-agnostic; no discretization details.
  - `algebra/symbolic/` — `SymbolicExpression`, `SymbolicOperator`,
    `AffineOperator`, custom AST nodes.
- **`discrete/core/`** — Abstract bases: `Discretization`, `DxOperators`,
  `DtOperators`, `BCTool`, `Domain`, `Boundary`, `DiscreteTime`.
- **`discrete/fd/`** — Finite-difference implementations: `FdDiscretization`,
  `FDOperator`, FD BCs, `FDDomain`, stencils.
- **`fieldspace/`** — User-facing facade. `FieldSpace` wires a `Discretization` to
  factories (`fields`, `dx`, `dt`, `systems`, `time`, `monitors`).
  `systems/les.py` assembles a `scipy.sparse.linalg.LinearOperator` and solves via
  CG.
- **`tools/`** — Dependency-free utilities:
  - `tools/symbolic/` — generic symbolic AST (`Symbolic`, `SymbolicNode`,
    `ValueNode`, `UnaryNode`, `BinaryNode`, `BinaryOpType`, `UnaryOpType`).
  - `tools/buffer/` — `ValueBuffer`, `DequeValueBuffer`, `ShiftProxyValueBuffer`.
  - `tools/geometry/` — `StructuredGridND`.
  - `tools/region/` — array region/slicing helpers.
  - `tools/advanceable/` — time-step advancement abstractions.
  - `tools/action.py` — `LazyAction`.

## Key entry points

| Class / symbol            | Location                                                 |
|---------------------------|----------------------------------------------------------|
| `Field`                   | `src/algebra/field.py:8`                                 |
| `Space` / `FieldShape`    | `src/algebra/space.py:7`, `src/algebra/space.py:22`      |
| `ShapeTransform`          | `src/algebra/space.py:15`                                |
| `Expression` (ABC)        | `src/algebra/expression.py:7`                            |
| `Operator` (ABC)          | `src/algebra/operator.py:11`                             |
| `SymbolicExpression`      | `src/algebra/symbolic/symbolic_expression.py:10`         |
| `SymbolicOperator`        | `src/algebra/symbolic/symbolic_operator.py:12`           |
| `AffineOperator`          | `src/algebra/symbolic/affine_operator.py:12`             |
| `Discretization` (ABC)    | `src/discrete/core/discretization.py:13`                 |
| `DxOperators` (ABC)       | `src/discrete/core/dx.py:5`                              |
| `DtOperators` (ABC)       | `src/discrete/core/dt.py:6`                              |
| `FdDiscretization`        | `src/discrete/fd/fd_discretization.py:12`                |
| `FDOperator`              | `src/discrete/fd/operators/core/fd_operator.py:10`       |
| `BoundaryCondition`/`BCType` | `src/discrete/core/bcs/bcs.py`                        |
| `BCTool` (ABC)            | `src/discrete/core/bcs/bc_tool.py:10`                    |
| `FieldSpace`              | `src/fieldspace/fieldspace.py:9`                         |
| `LES`                     | `src/fieldspace/systems/les.py:11`                       |
| `ShapeMismatchError`      | `src/algebra/exceptions.py:1`                            |

## Typical user flow

Reference: `tests/scripts/script_test_diffusion.py`.

1. Build a discretization: `grid = StructuredGridND(...)`; `discrete =
   fd.FdDiscretization(grid)`.
2. Wrap in a `FieldSpace`: `s = FieldSpace(discrete)`. Exposes `s.fields`, `s.dx`,
   `s.dt`, `s.systems`, `s.time`, `s.monitors`.
3. Create fields via `s.fields.scalar() / vector() / tensor()` (returns `Field`
   backed by a `DequeValueBuffer`).
4. Compose operators symbolically: `f_dx = s.dx.laplace()`,
   `f_dt = s.dt.euler(F)` (returns an `AffineOperator`); then
   `lhs = f_dt - L * f_dx`. Algebra is overloaded via `BinaryOpType` and
   `Symbolic` AST nodes — purely symbolic, no numerics yet.
5. Form an equation: `equation = s.systems.les(lhs, rhs.value(), bcs)` where BCs
   come from `s.systems.bc.dirichlet(...)` / `neumann(...)`. `LES` separates the
   affine bias from `lhs` into `rhs`.
6. Solve / advance: inside `s.time.run(...)`,
   `solution = equation.solve()` returns a `SymbolicExpression` whose `.eval()`
   invokes CG (`LES._assemble` builds the `LinearOperator`; BCs are applied via
   `BCTool`). Then `F.set_value(solution).perform()` (a `LazyAction`) writes the
   result into the field buffer.

The user never touches arrays directly; algorithms are written symbolically
against `Operator` / `Expression` / `Field`, and the `Discretization` supplies
the concrete operators used during `eval()` / `apply()`.

## Architectural rules

- `algebra` MUST NOT import from `discrete`.
- `discrete` may depend on `algebra` and `tools` only.
- `fieldspace` is the only layer that wires `discrete` + `algebra` for users.
- `tools` stays dependency-free of `algebra`, `discrete`, and `fieldspace`.
- User-facing APIs operate on `Field`s, not raw `np.ndarray`.
- State mutations are returned as `LazyAction`; the caller is responsible for
  `.perform()`.

## Conventions

### Type hints

All public inputs and returns are fully type-hinted. Use
`from __future__ import annotations`, `Self`, `TypeVar`, and `Generic` where
appropriate. Docstrings may be omitted on small methods; larger classes should
include them.

### Naming

- Classes: `PascalCase`; modules: `snake_case`.
- FD-specific classes are prefixed `FD` (e.g., `FdDiscretization`, `FDOperator`,
  `FDDxOperators`, `FDBCTool`, `FDDomain`).
- Abstract bases live in `core/` subpackages; concrete implementations live in
  sibling packages (e.g., `discrete/core/` vs `discrete/fd/`).
- Type aliases such as `TOperator`, `TBoundary`, `TDomain`, `TSymbolic` are used
  with `Generic` for typed abstracts.

### Symbolic design

Operator/Expression algebra is built on `Symbolic[T]`
(`src/tools/symbolic/symbolic.py:9`) using frozen-dataclass node trees
(`ValueNode` / `UnaryNode` / `BinaryNode`). Resolution is lazy via `.resolve()` /
`.eval()`. `_compatible()` hooks gate which combinations are allowed (per shape,
space, `ShapeTransform`).

### Other

- Frozen dataclasses are used for value-like objects (`Space`, `FieldShape`,
  `BoundaryCondition`, symbolic nodes).
- The only custom exception is `ShapeMismatchError`; other validation uses
  `assert` and `ValueError`.
- `__init__.py` re-exports only the small public surface of each subpackage.

## Tests

Layout mirrors `src/` (`pytest.ini` sets `pythonpath = src`).

- `tests/algebra/` — unit tests for the algebra/symbolic layer.
  - `tests/algebra/conftest.py` provides a `MockOperator` test double.
  - `tests/algebra/symbolic/` — `test_symbolic_expression.py`,
    `test_symbolic_operator.py`, `test_affine_operator.py`.
- `tests/discrete/fd/` — FD-specific unit tests.
  - `tests/discrete/fd/operators/` — `test_grad.py`, `test_div.py`,
    `test_lap.py`.
  - `tests/discrete/fd/stencil/` — `test_stencil.py`, `test_ax_stencil.py`.
  - `tests/discrete/fd/test_ders.py`.
- `tests/tools/` — `test_symbolic.py` (generic symbolic),
  `tests/tools/region/` (`test_region.py`, `test_region_utils.py`).
- `tests/integration/` — `test_les.py` (collected); `script_test_dirichlet.py`,
  `script_test_neumann.py` (manual scripts, not collected).
- `tests/scripts/` — runnable demos (`script_test_diffusion.py`,
  `script_test_laplace.py`, `script_test_poisson.py`,
  `script_test_navier_stokes.py`); used as end-to-end usage examples.

Pytest collects `test_*.py`; demo/integration scripts use the `script_test_*.py`
prefix to opt out of collection.
