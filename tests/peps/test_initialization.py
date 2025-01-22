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
""" Test Peps state initialization. """
import pytest
import yastn
import yastn.tn.fpeps as fpeps

tol = 1e-12  #pylint: disable=invalid-name


def test_propuct_peps(config_kwargs):
    """ Generate a product peps on few lattices. Check exceptions. """
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)

    geometry = fpeps.CheckerboardLattice()
    v0 = ops.vec_n(val=0)
    v1 = ops.vec_n(val=1)
    psi = fpeps.product_peps(geometry, {(0, 0): v0, (0, 1): v1})
    for site in psi.sites():
        legs = psi[site].get_legs()
        for leg in legs:
            assert leg.t == ((0,),)
    #
    #
    geometry = fpeps.SquareLattice(dims=(3, 2), boundary='obc')
    psi = fpeps.product_peps(geometry, ops.I())
    for site in psi.sites():
        legs = psi[site].get_legs()
        for leg in legs:
            assert leg.t == ((0,),)
    #
    #
    with pytest.raises(yastn.YastnError):
        psi = fpeps.product_peps(geometry, {(0, 0): v0})
        # product_peps did not initialize some peps tensor

    with pytest.raises(yastn.YastnError):
        fpeps.product_peps(geometry, ops.I().add_leg(s=1))
        # Some vector has ndim > 2

    with pytest.raises(yastn.YastnError):
        fpeps.product_peps(psi, v0)
        # Geometry should be an instance of SquareLattice or CheckerboardLattice


def test_save_load_copy(config_kwargs):
    geometries = [fpeps.SquareLattice(dims=(3, 2), boundary='obc'),
                  fpeps.CheckerboardLattice()]

    config = yastn.make_config(sym='none', **config_kwargs)
    for geometry in geometries:
        vecs = {site: yastn.rand(config, s=(1,), D=(3,)) for site in geometry.sites()}
        psi = fpeps.product_peps(geometry, vecs)

        d = psi.save_to_dict()
        psi2 = fpeps.load_from_dict(config, d)
        psi3 = psi.copy()
        psi4 = psi.clone()

        for site in psi.sites():
            for phi in [psi2, psi3, psi4]:
                assert (phi[site] - psi[site]).norm() < tol


def test_Peps_initialization(config_kwargs):
    geometry = fpeps.CheckerboardLattice()
    psi = fpeps.Peps(geometry)

    config = yastn.make_config(sym='U1', **config_kwargs)
    leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
    #
    # for rank-5 tensors
    #
    A00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj(), leg])
    A01 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj(), leg])
    psi[0, 0] = A00
    # Currently, 5-leg PEPS tensors are fused by __setitem__ as ((top-left)(bottom-right) physical).
    # This is done to work with object having smaller number of blocks.
    assert psi[0, 0].ndim == 3
    assert (psi[0, 0].unfuse_legs(axes=(0, 1)) - A00).norm() < 1e-13

    # PEPS with no physical legs is also possible.
    #
    # for rank-4 tensors
    #
    B00 = yastn.rand(config, legs=[leg.conj(), leg, leg, leg.conj()])
    psi[0, 0] = B00
    assert psi[0, 0].ndim == 4

    # PEPS with tensors assigned during initialization
    #
    psi = fpeps.Peps(geometry, tensors={(0, 0): A00, (0, 1): A01})
    assert (psi[0, 0].unfuse_legs(axes=(0, 1)) - A00).norm() < 1e-13
    assert (psi[1, 1].unfuse_legs(axes=(0, 1)) - A00).norm() < 1e-13
    assert (psi[0, 1].unfuse_legs(axes=(0, 1)) - A01).norm() < 1e-13
    assert (psi[1, 0].unfuse_legs(axes=(0, 1)) - A01).norm() < 1e-13
    #
    # or equivalently (with some provided tensors being redundant)
    #
    psi = fpeps.Peps(geometry, tensors=[[A00, A01], [A01, A00]])
    assert (psi[0, 0].unfuse_legs(axes=(0, 1)) - A00).norm() < 1e-13
    assert (psi[1, 0].unfuse_legs(axes=(0, 1)) - A01).norm() < 1e-13
    #
    # raising exceptions
    with pytest.raises(yastn.YastnError):
        fpeps.Peps(geometry, tensors={(0, 0): A00, (1, 1): A01})
        # Peps: Non-unique assignment of tensor to unique lattice sites.
    with pytest.raises(yastn.YastnError):
        g_finite = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
        fpeps.Peps(g_finite, tensors={(0, 4): A00})
        # Peps: tensors assigned outside of the lattice geometry.
    with pytest.raises(yastn.YastnError):
        fpeps.Peps(geometry, tensors={0: A00})
        # Peps: Non-unique assignment of tensor to unique lattice sites.
    with pytest.raises(yastn.YastnError):
        fpeps.Peps(geometry, tensors=[A00])
        # Peps: tensors assigned outside of the lattice geometry.
    with pytest.raises(yastn.YastnError):
        fpeps.Peps(geometry, tensors=[[A00]])
        # Peps: Not all unique lattice sites got assigned with a tensor.


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
