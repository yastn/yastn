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

try:
    from .configs import config as cfg
    # cfg is used by pytest to inject different backends and divices
except ImportError:
    from configs import config as cfg


tol = 1e-12

def test_propuct_peps():
    """ Generate a few lattices veryfing expected output of some functions. """
    ops = yastn.operators.SpinlessFermions(sym='U1', backend=cfg.backend, default_device=cfg.default_device)

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


def test_save_load_copy():
    geometries = [fpeps.SquareLattice(dims=(3, 2), boundary='obc'),
                  fpeps.CheckerboardLattice()]

    for geometry in geometries:
        vecs = {site: yastn.rand(cfg, s=(1,), D=(3,)) for site in geometry.sites()}
        psi = fpeps.product_peps(geometry, vecs)

        d = psi.save_to_dict()
        psi2 = fpeps.load_from_dict(cfg, d)
        psi3 = psi.copy()
        psi4 = psi.clone()

        for site in psi.sites():
            for phi in [psi2, psi3, psi4]:
                assert (phi[site] - psi[site]).norm() < tol


if __name__ == '__main__':
    test_propuct_peps()
    test_save_load_copy()
