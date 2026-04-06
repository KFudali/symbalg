import pytest
from tools import Symbolic




def test_symbolic_fold_single_value():
    sym = Symbolic[int](5)
    assert sym.fold() == 5

def test_symbolic_fold_int_sum():
    sym = Symbolic[int](1)
    sym += 1
    assert sym.fold() == 2
