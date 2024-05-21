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
""" basic methods of single Mps """
import pytest
import yastn
import yastn.tn.mps as mps
try:
    from .configs import config_dense as cfg
except ImportError:
    from configs import config_dense as cfg
# pytest modifies cfg to inject different backends and divices during tests


def test_canonize(config=cfg, tol=1e-12):
    """ Initialize random mps and checks canonization. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}
    N = 16

    ops = yastn.operators.Spin1(sym='Z3', **opts_config)
    I = mps.product_mpo(ops.I(), N=N)
    for n in (0, 1, 2):
        psi = mps.random_mps(I, n=n, D_total=16)
        check_canonize(psi, tol)
    H = mps.random_mpo(I, D_total=8, dtype='complex128')
    check_canonize(H, tol)

    ops = yastn.operators.Spin12(sym='dense', **opts_config)
    psi = mps.random_mps(I, D_total=16, dtype='complex128')
    check_canonize(psi, tol)
    H = mps.random_mpo(I, D_total=8)
    check_canonize(H, tol)

    with pytest.raises(yastn.YastnError):
        psi.orthogonalize_site_(4, to="center")
        # "to" should be in "first" or "last"
    with pytest.raises(yastn.YastnError):
        psi.orthogonalize_site_(4, to="last")
        psi.orthogonalize_site_(5, to="last")
        # Only one central block is allowed. Attach the existing central block before orthogonalizing site.


def check_canonize(psi, tol):
    """ Canonize mps to left and right, running tests if it is canonical. """
    ref_s = (-1, 1, 1) if psi.nr_phys == 1 else (-1, 1, 1, -1)
    norm = psi.norm()
    for to in ('last', 'first'):
        psi.canonize_(to=to, normalize=False)
        assert psi.is_canonical(to=to, tol=tol)
        assert all(psi[site].s == ref_s for site in psi.sweep())
        assert psi.pC is None
        assert len(psi.A) == len(psi)
        assert abs(psi.factor / norm - 1) < tol
        assert abs(mps.vdot(psi, psi) / norm ** 2 - 1) < tol

    for to in ('last', 'first'):
        phi = psi.shallow_copy()
        phi.canonize_(to=to)
        assert abs(phi.factor - 1) < tol
        assert abs(mps.vdot(phi, phi) - 1) < tol


def test_reverse(config=cfg, tol=1e-12):
    """ Initialize random mps and checks canonization. """
    opts_config = {} if config is None else \
                {'backend': config.backend, 'default_device': config.default_device}

    N = 8
    ops = yastn.operators.Spin1(sym='Z3', **opts_config)
    I = mps.product_mpo(ops.I(), N=N)

    psi = mps.random_mps(I, n=2, D_total=16).canonize_(to='first')
    psi.orthogonalize_site_(n=0, to='last', normalize=False)

    phi = psi.reverse_sites()
    phi.absorb_central_(to='last')
    phi = phi.reverse_sites()

    psi.absorb_central_(to='last')
    assert abs(mps.vdot(phi, psi) - 1) < tol


    psi = mps.random_mpo(I, D_total=8)
    phi = psi.reverse_sites()
    phi = phi.reverse_sites()
    assert abs(mps.vdot(phi, psi) / mps.vdot(phi, phi) - 1) < tol


if __name__ == "__main__":
    test_canonize()
    test_reverse()
