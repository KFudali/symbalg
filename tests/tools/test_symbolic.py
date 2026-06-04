import numpy as np
from tools.symbolic import Symbolic


def test_symbolic_fold_single_value():
    sym = Symbolic[int].wrap(5)
    assert sym.resolve() == 5


def test_symbolic_fold_int_sum():
    sym = Symbolic[int].wrap(1)
    sym = sym + 1
    assert sym.resolve() == 2

    sym = sym + 3
    assert sym.resolve() == 5

    a = Symbolic[float].wrap(1.0)
    b = Symbolic[float].wrap(5.0)
    c = Symbolic[float].wrap(10.0)
    d = Symbolic[float].wrap(20.0)
    assert np.isclose(((a * b - c) / d).resolve(), -0.25)
