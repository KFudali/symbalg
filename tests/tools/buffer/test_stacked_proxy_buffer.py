import numpy as np
from tools.buffer import StackedProxyValueBuffer, ConstValueBuffer


def test_stacked_proxy_buffer():
    ones = ConstValueBuffer(np.ones((10, 10), dtype=float))
    twos = ConstValueBuffer(np.ones((10, 10), dtype=float) * 2)
    stacked = StackedProxyValueBuffer((ones, twos))

    assert stacked.shape == (2, 10, 10)
    assert np.allclose(stacked.get()[0], 1.0)
    assert np.allclose(stacked.get()[1], 2.0)

    ones.set(np.ones((10, 10)) * 3.0)
    twos.set(np.ones((10, 10)) * 5.0)

    assert stacked.shape == (2, 10, 10)
    assert np.allclose(stacked.get()[0], 3.0)
    assert np.allclose(stacked.get()[1], 5.0)
