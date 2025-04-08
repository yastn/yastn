# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" yastn.conj() yastn.flip_signature() """
import numpy as np
import pytest
import yastn

tol = 1e-12  #pylint: disable=invalid-name


def conj_vs_numpy(a, expected_n):
    """ run conj(), flip_signature() and a few tests. """
    b = a.conj()
    c = a.flip_signature()
    d = a.conj_blocks()
    assert all(x.is_consistent() for x in (b, c, d))

    assert all(x.struct.n == expected_n for x in (b, c))
    assert a.struct.n == d.struct.n

    assert all(sa + sb == 0 for sa, sb in zip(a.struct.s, b.struct.s))
    assert all(sa + sc == 0 for sa, sc in zip(a.struct.s, c.struct.s))
    assert a.struct.s == d.struct.s

    na, nb, nc, nd = a.to_numpy(), b.to_numpy(), c.to_numpy(), d.to_numpy()
    assert np.linalg.norm(na.conj() - nb) < tol
    assert np.linalg.norm(na - nc) < tol
    assert np.linalg.norm(na.conj() - nd) < tol
    assert abs(yastn.vdot(a, a).item() - yastn.vdot(b, a, conj=(0, 0)).item()) < tol
    assert abs(yastn.vdot(a, a).item() - yastn.vdot(d, c, conj=(0, 0)).item()) < tol

    assert yastn.norm(a - b.conj()) < tol
    assert yastn.norm(a - c.flip_signature()) < tol
    assert yastn.norm(a - d.conj_blocks()) < tol


def test_conj_basic(config_kwargs):
    """ test conj for different symmerties """
    # Z2
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    a = yastn.randC(config=config_Z2, s=(1, 1, 1, -1, -1, -1), n=1,
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    conj_vs_numpy(a, expected_n=(1,))

    # Z2xU1
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    a = yastn.rand(config=config_Z2xU1, s=(1, -1), n=(1, 2),
                  t=[[(0, 0), (1, 1), (0, 2)], [(0, 1), (0, 0), (1, 1)]],
                  D=[[1, 2, 3], [4, 5, 6]])
    conj_vs_numpy(a, expected_n=(1, -2))


def test_conj_hard_fusion(config_kwargs):
    config_Z2 = yastn.make_config(sym='Z2', **config_kwargs)
    a = yastn.randC(config=config_Z2, s=(1, -1, 1, -1, 1, -1),
                  t=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
                  D=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    a = a.fuse_legs(axes=((0, 1), (2, 3), (4, 5)))
    a = a.fuse_legs(axes=((0, 1), 2))
    b = a.conj()
    c = a.flip_signature()
    d = a.conj_blocks()
    assert all(sa + sb == 0 for sa, sb in zip(a.struct.s, b.struct.s))
    assert all(sa + sc == 0 for sa, sc in zip(a.struct.s, c.struct.s))
    assert a.struct.s == d.struct.s

    assert all(sa + sb == 0 for hfa, hfb in zip(a.hfs, b.hfs) for sa, sb in zip(hfa.s, hfb.s))
    assert all(sa + sc == 0 for hfa, hfc in zip(a.hfs, c.hfs) for sa, sc in zip(hfa.s, hfc.s))
    assert all(hfa.s == hfd.s for hfa, hfd in zip(a.hfs, d.hfs))


def test_flip_charges(config_kwargs):
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, 1), (1, 0), (0, -1)), D=(2, 3, 2))
    a = yastn.rand(config=config_Z2xU1, legs=[leg, leg, leg.conj(), leg.conj()])
    b = a.flip_charges()
    c = a.flip_charges(axes=(1, 2))

    assert a.s == (1, 1, -1, -1)
    assert b.s == (-1, -1, 1, 1)
    assert c.s == (1, -1, 1, -1)
    assert b.get_legs() == (leg.conj(), leg.conj(), leg, leg)
    assert c.get_legs() == (leg, leg.conj(), leg, leg.conj())
    assert all(x.is_consistent() for x in (a, b, c))
    assert all(yastn.are_independent(a, x) for x in (b, c))
    assert (a - b.conj()).norm() > tol
    assert (a - b.flip_charges()).norm() < tol
    assert (a - c.flip_charges(axes=(2, 1))).norm() < tol

    with pytest.raises(yastn.YastnError):
        d = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
        d.flip_charges(axes=1)
        # Flipping charges of hard-fused leg is not supported.
    with pytest.raises(yastn.YastnError):
        d = yastn.rand(config_Z2xU1, legs=leg, isdiag=True)
        d.flip_charges()
        # Cannot flip charges of a diagonal tensor. Use diag() first.


def test_switch_signature(config_kwargs):
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    leg = yastn.Leg(config_Z2xU1, s=1, t=((0, 1), (1, 0), (0, -1)), D=(2, 3, 2))
    a = yastn.rand(config=config_Z2xU1, legs=[leg, leg, leg.conj(), leg.conj()])
    b = a.switch_signature(axes='all')
    c = a.switch_signature(axes=(1, 2))

    assert a.s == (1, 1, -1, -1)
    assert b.s == (-1, -1, 1, 1)
    assert c.s == (1, -1, 1, -1)
    assert b.get_legs() == (leg.conj(), leg.conj(), leg, leg)
    assert c.get_legs() == (leg, leg.conj(), leg, leg.conj())
    assert all(x.is_consistent() for x in (a, b, c))
    assert all(yastn.are_independent(a, x) for x in (b, c))
    assert (a - b.conj()).norm() > tol
    assert (a - b.switch_signature(axes='all')).norm() < tol
    assert (a - c.switch_signature(axes=(2, 1))).norm() < tol

    d = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
    e= d.switch_signature(axes=1)
    d0= e.switch_signature(axes=1)
    assert d0.get_legs() == d.get_legs()
    assert (d0 - d).norm() < tol
    a0= d0.unfuse_legs(axes=1)
    assert a0.get_legs() == a.get_legs()
    assert (d0 - d).norm() < tol
    assert a.switch_signature(axes=(1,2)).get_legs() == e.unfuse_legs(axes=1).get_legs()
    with pytest.raises(AssertionError):
        assert (a.switch_signature(axes=(1,2)) - e.unfuse_legs(axes=1)).norm() < tol
    # switch_signature supports hard-fused legs.
    
    with pytest.raises(yastn.YastnError):
        f = yastn.rand(config_Z2xU1, legs=leg, isdiag=True)
        f.switch_signature(axes='all')
        # Cannot flip charges of a diagonal tensor. Use diag() first.



def test_conj_Z2xU1(config_kwargs):
    #
    # create random complex-valued symmetric tensor with symmetry Z2 x U1
    #
    config_Z2xU1 = yastn.make_config(sym=yastn.sym.sym_Z2xU1, **config_kwargs)
    legs = [yastn.Leg(config_Z2xU1, s=1, t=((0, 2), (1, 1), (1, 2)), D=(1, 2, 3)),
            yastn.Leg(config_Z2xU1, s=-1, t=((0, 0), (0, -1), (1, 0)), D=(4, 5, 6))]
    a = yastn.rand(config=config_Z2xU1, legs=legs, n=(1, 2), dtype="complex128")
    #
    # conjugate tensor a: verify that signature and total charge
    # has been reversed.
    #
    b = a.conj()
    assert b.get_tensor_charge() == (1, -2)
    assert b.get_signature() == (-1, 1)

    #
    # Interpreting these tensors a,b as vectors, following contraction
    #  _           _
    # | |-<-0 0-<-| |
    # |a|->-1 1->-|b|
    #
    # is equivalent to computing the square of Frobenius norm of a.
    # Result is chargeless single-element tensor equivalent to scalar.
    #
    norm_F = yastn.tensordot(a, b, axes=((0, 1), (0, 1)))
    assert norm_F.get_tensor_charge() == (0, 0)
    assert norm_F.get_signature() == ()
    assert abs(a.norm()**2 - norm_F.real().to_number()) < 1e-12

    #
    # only complex-conjugate elements of the blocks of tensor a, leaving
    # the structure i.e. signature and total charge intact.
    #
    c = a.conj_blocks()
    assert c.get_tensor_charge() == a.get_tensor_charge()
    assert c.get_signature() == a.get_signature()

    #
    # flip signature of the tensor c and its total charge, but do not
    # complex-conjugate elements of its block
    #
    d = c.flip_signature()
    assert d.get_tensor_charge() == b.get_tensor_charge()
    assert d.get_signature() == b.get_signature()

    #
    # conj() is equivalent to flip_signature().conj_blocks() (or in the
    # opposite order). Hence, tensor b and tensor d should be numerically
    # identical
    #
    assert yastn.norm(b - d) < 1e-12


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])

