"""
to_nonsymmetric()  to_dense()  to_numpy()
"""
import numpy as np
import pytest
import yast
try:
    from .configs import config_U1
except ImportError:
    from configs import config_U1

tol = 1e-12  #pylint: disable=invalid-name


# def _test_embed(a, dict_legs, a_shape, new_shape):
#     a2 = a.embed(legs=dict_legs)
#     assert yast.norm(a - a2) < tol  # == 0.
#     assert a.get_shape() == a_shape
#     assert a2.get_shape() == new_shape
#     assert all(x.is_consistent() for x in (a, a2))
#     assert yast.are_independent(a, a2)
#     assert a2.embed(legs=dict(enumerate(a.get_legs()))) is a2


# def test_embed_basic():
#     """ a.embed(legs=...)"""
#     a = yast.rand(config=config_U1, s=(-1, -1, 1),
#                       t=((-1, 1, 2), (-1, 1, 2), (1, 2)),
#                       D=((1, 3, 4), (4, 5, 6), (3, 4)))
#     b = yast.rand(config=config_U1, s=(-1, -1, 1),
#                       t=((-1, 0), (1, 2), (0, 1)),
#                       D=((1, 2), (5, 6), (2, 3)))
    
#     assert a.embed() is a

#     b_legs = dict(enumerate(b.get_legs()))
#     _test_embed(a, b_legs, a_shape=(8, 15, 7), new_shape=(10, 15, 9))

#     fa = a.fuse_legs(axes=((0, 1), 2), mode='meta')
#     fb = b.fuse_legs(axes=((0, 1), 2), mode='meta')
#     fb_legs = dict(enumerate(fb.get_legs()))
#     _test_embed(fa, fb_legs, a_shape=(37, 7), new_shape=(76, 9))

#     ha = a.fuse_legs(axes=((0, 1), 2), mode='hard')
#     hb = b.fuse_legs(axes=((0, 1), 2), mode='hard')
#     hb_legs = dict(enumerate(hb.get_legs()))
#     _test_embed(ha, hb_legs, a_shape=(37, 7), new_shape=(76, 9))

#     ffa = fa.fuse_legs(axes=[(0, 1)], mode='meta')
#     ffb = fb.fuse_legs(axes=[(0, 1)], mode='meta')
#     ffb_legs = dict(enumerate(ffb.get_legs()))
#     _test_embed(ffa, ffb_legs, a_shape=(126,), new_shape=(238,))

#     hha = ha.fuse_legs(axes=[(0, 1)], mode='hard')
#     hhb = hb.fuse_legs(axes=[(0, 1)], mode='hard')
#     hhb_legs = dict(enumerate(hhb.get_legs()))
#     _test_embed(hha, hhb_legs, a_shape=(126,), new_shape=(238,))

#     fha = ha.fuse_legs(axes=[(0, 1)], mode='meta')
#     fhb = hb.fuse_legs(axes=[(0, 1)], mode='meta')
#     fhb_legs = dict(enumerate(fhb.get_legs()))
#     _test_embed(fha, fhb_legs, a_shape=(126,), new_shape=(238,))


if __name__ == '__main__':
    pass
    # test_embed_basic()
