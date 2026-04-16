# Code Conventions

USE VENV: ~/.virtualenvs/algfields3.12/

## Style & Formatting

- **Linter:** pylint
- **Formatter:** black
- **Testing:** pytest

## Type Hints

All input and return types must be type hinted. Method docstrings can be omitted, but larger classes should include them.

## Architecture

This package provides a lightweight abstraction layer for algebra on fields. The base type is `np.ndarray`.
