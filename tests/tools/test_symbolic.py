import pytest
from tools.symbolic import Symbolic
import numpy

def test_symbolic_fold_single_value():
    sym = Symbolic[int](5)
    assert sym.fold() == 5

def test_symbolic_fold_int_sum():
    sym = Symbolic[int](1)
    sym += 1
    assert sym.fold() == 2

    sym += 3
    assert sym.fold() == 5

def test_symbolic_fold_int_sum():
    a = Symbolic[int](1.0)
    b = Symbolic[int](5.0)
    c = Symbolic[int](10.0)
    d = Symbolic[int](20.0)
    assert numpy.isclose(
        ((a*b - c)/d).fold(), -0.25
    )