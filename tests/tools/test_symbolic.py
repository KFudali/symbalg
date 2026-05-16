import numpy as np
from tools.symbolic import Symbolic


def test_symbolic_fold_single_value():
    sym = Symbolic[int](5)
    assert sym.resolve() == 5


def test_symbolic_fold_int_sum():
    sym = Symbolic[int](1)
    sym = sym + 1
    assert sym.resolve() == 2

    sym = sym + 3
    assert sym.resolve() == 5

    a = Symbolic[int](1.0)
    b = Symbolic[int](5.0)
    c = Symbolic[int](10.0)
    d = Symbolic[int](20.0)
    assert np.isclose(((a * b - c) / d).resolve(), -0.25)
