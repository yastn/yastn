""" Test elements of fuse_legs(... mode='hard') """
import numpy as np
import pytest
import yast
try:
    from .configs import config_dense
except ImportError:
    from configs import config_dense

tol = 1e-10  #pylint: disable=invalid-name

# example test for dense merging hard
def test_hard_trace():
    a = yast.rand(config=config_dense, s=(-1, 1, 1, -1), D=(3, 2, 6, 4), dtype='float64')
    af = yast.fuse_legs(a, axes=((1, 2), (3, 0)), mode='hard')
    tra = yast.trace(a, axes=((1, 2), (3, 0)))
    traf = yast.trace(af, axes=(0, 1))
    assert yast.norm_diff(tra, traf) < tol

if __name__ == '__main__':
    test_hard_trace()