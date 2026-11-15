# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" MpsMpoOBC.to_nonsymmetric() """
import pytest
import yastn
import yastn.tn.mps as mps

tol = 1e-12  # pylint: disable=invalid-name


def _heisenberg(ops, N):
    I = mps.product_mpo(ops.I(), N)  # noqa: E741
    terms = [mps.Hterm(0.5, [i, i + 1], [ops.sp(), ops.sm()]) for i in range(N - 1)]
    terms += [mps.Hterm(0.5, [i, i + 1], [ops.sm(), ops.sp()]) for i in range(N - 1)]
    terms += [mps.Hterm(1.0, [i, i + 1], [ops.sz(), ops.sz()]) for i in range(N - 1)]
    return I, mps.generate_mpo(I, terms)


@pytest.mark.parametrize("key", [None, "reverse"])
def test_mps_mpo_to_nonsymmetric(config_kwargs, key):
    """ Overlaps and expectation values are the same after converting MPS and MPO. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    ops.random_seed(seed=0)
    I, H = _heisenberg(ops, N=6)  # noqa: E741
    psi = mps.random_mps(I, n=0, D_total=8)
    psi.canonize_(to='first')  # canonical form makes the legs of neighboring tensors consistent

    psi_ns, H_ns = psi.to_nonsymmetric(key=key), H.to_nonsymmetric(key=key)
    assert all(x[n].config.sym.SYM_ID == 'dense' for x in (psi_ns, H_ns) for n in x.sweep())
    assert abs(mps.vdot(psi_ns, psi_ns) - mps.vdot(psi, psi)) < tol
    assert abs(mps.vdot(psi_ns, H_ns, psi_ns) - mps.vdot(psi, H, psi)) < tol
    assert psi.config.sym.SYM_ID == 'U1'  # the original is not modified

    # central block and factor
    phi = psi.shallow_copy()
    phi.orthogonalize_site_(2, to='last', normalize=False)
    phi_ns = phi.to_nonsymmetric(key=key)
    assert phi_ns.pC == phi.pC and phi_ns.factor == phi.factor
    assert phi_ns.A[phi_ns.pC].config.sym.SYM_ID == 'dense'
    assert abs(mps.vdot(phi_ns, phi_ns) - mps.vdot(phi, phi)) < tol


def test_mps_to_nonsymmetric_fused(config_kwargs):
    """ Hard-fused physical legs keep their fusion; fused virtual legs raise an error. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    ops.random_seed(seed=0)
    _, H = _heisenberg(ops, N=4)
    # MPO legs are (left, ket, right, bra); fused ket and bra legs give MPS with a hard-fused physical leg
    phi = mps.Mps(N=H.N)
    for n in H.sweep():
        phi[n] = H[n].fuse_legs(axes=(0, (1, 3), 2), mode='hard')
    phi_ns = phi.to_nonsymmetric()
    assert all(phi_ns[n].get_legs(axes=1).is_fused() for n in phi_ns.sweep())
    assert abs(mps.vdot(phi_ns, phi_ns) - mps.vdot(phi, phi)) < tol * abs(mps.vdot(phi, phi))
    H_ns = H.to_nonsymmetric()
    assert all((phi_ns[n].unfuse_legs(axes=1) - H_ns[n].transpose(axes=(0, 1, 3, 2))).norm() < tol
               for n in phi.sweep())

    # MPO: a hard-fused bra leg (axis 3) is a physical leg
    Hf = H.shallow_copy()
    for n in H.sweep():
        Hf[n] = H[n].add_leg(axis=4, s=-1).fuse_legs(axes=(0, 1, 2, (3, 4)), mode='hard')
    Hf_ns = Hf.to_nonsymmetric()
    assert all(Hf_ns[n].get_legs(axes=3).is_fused() for n in Hf_ns.sweep())

    # a fused virtual leg is not supported (MPO: right virtual leg is axis 2)
    Hv = H.shallow_copy()
    Hv.A[0] = H[0].add_leg(axis=3, s=-1).fuse_legs(axes=(0, 1, (2, 3), 4), mode='hard')
    with pytest.raises(yastn.YastnError,
                       match="to_nonsymmetric\\(\\) is not supported for fused virtual legs."):
        Hv.to_nonsymmetric()


def test_mps_to_nonsymmetric_fermionic(config_kwargs):
    """ Swap gates act trivially on non-symmetric tensors; fermionic MPS raises an error. """
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    I = mps.product_mpo(ops.I(), N=4)  # noqa: E741
    psi = mps.random_mps(I, n=2, D_total=4)
    with pytest.raises(yastn.YastnError,
                       match="to_nonsymmetric\\(\\) is not supported for fermionic config"):
        psi.to_nonsymmetric()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
