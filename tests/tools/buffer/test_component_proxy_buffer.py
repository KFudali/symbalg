import numpy as np
from tools.buffer import ComponentProxyValueBuffer, DequeValueBuffer


def test_component_proxy_single_shapes():
    source = DequeValueBuffer(shape=(3, 10, 10))
    zeros = np.zeros(dtype=float, shape=(10, 10))
    ones = np.ones(dtype=float, shape=(10, 10))
    twos = np.ones(dtype=float, shape=(10, 10)) * 2.0

    source.set(np.stack((zeros, ones, twos)))
    zero_comp = ComponentProxyValueBuffer(source, (slice(0, 1), slice(None)))
    one_comp = ComponentProxyValueBuffer(source, (slice(1, 2), slice(None)))
    two_comp = ComponentProxyValueBuffer(source, (slice(2, 3), slice(None)))
    comps = (zero_comp, one_comp, two_comp)
    for comp in comps:
        assert comp.shape == (10, 10)
    assert np.array_equal(zero_comp.get(), zeros)
    assert np.array_equal(one_comp.get(), ones)
    assert np.array_equal(two_comp.get(), twos)
