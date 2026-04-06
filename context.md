# Code Conventions

## Style & Formatting

- **Linter:** pylint
- **Formatter:** black
- **Testing:** pytest

## Type Hints

All input and return types must be type hinted. Method docstrings can be omitted, but larger classes should include them.

## Architecture

This package provides a lightweight abstraction layer for algebra on fields. The base type is `np.ndarray`.



Sketch of the architecture
algebra/
    operator/
        operator.py
        callable_operator.py
    expression/
        expression.py
        callable_expression.py
    symbolic/
        symbolic_operator.py
        symbolic_expression.py
    systems/
        eq_system.py
        les.py
        bcs/
            dirichlet.py
            neumann.py
    space/
        space.py
        domain/
            boundary/
                boundary_id.py
                boundary.py
            subdomain/
                subdomain.py
                subdomain_id.py
tools/
    stencil.py
    region.py
    buffers/
        value_buffer.py
space/
    field/
    time/
        
discrete/
    core/
        discretization.py
        discrete_operator.py
        discrete_bc.py
    fd/
        discretization.py
        operators/
            stencil_operator.py
            laplace.py
            grad.py
            div.py
