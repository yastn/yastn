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
""" to_nonsymmetric() of Peps, EnvCTM, and EnvBoundaryMPS """
import pytest
import yastn
import yastn.tn.fpeps as fpeps
import yastn.tn.mps as mps

tol = 1e-12  # pylint: disable=invalid-name


def _random_peps(ops, dims, boundary, seed=0):
    """ Random U(1) PEPS with consistent legs; trivial legs at the open boundaries. """
    ops.config.backend.random_seed(seed=seed)
    g = fpeps.SquareLattice(dims=dims, boundary=boundary)
    v = yastn.Leg(ops.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    one = yastn.Leg(ops.config, s=1, t=(0,), D=(1,))
    obc = boundary == 'obc'
    tensors = {}
    for site in g.sites():
        t = one if obc and site[0] == 0 else v
        l = one if obc and site[1] == 0 else v  # noqa: E741
        b = one if obc and site[0] == dims[0] - 1 else v
        r = one if obc and site[1] == dims[1] - 1 else v
        A = yastn.rand(config=ops.config, legs=[t.conj(), l, b, r.conj(), ops.space()], n=0)
        tensors[site] = A / A.norm()
    return fpeps.Peps(g, tensors=tensors)


def _dense_ops(ops, key=None):
    """ Spin operators converted with the full physical leg, as some of them do not carry all charges. """
    legs = {0: ops.space(), 1: ops.space().conj()}
    return {name: getattr(ops, name)().to_nonsymmetric(legs=legs, key=key) for name in ('I', 'sz', 'sp', 'sm')}


def test_peps_to_nonsymmetric(config_kwargs):
    """ Converted finite PEPS represents the same state. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    psi = _random_peps(ops, dims=(2, 3), boundary='obc')
    psi_ns = psi.to_nonsymmetric()
    assert all(psi_ns[site].config.sym.SYM_ID == 'dense' for site in psi_ns.sites())
    assert psi[psi.sites()[0]].config.sym.SYM_ID == 'U1'  # the original is not modified
    assert abs(psi_ns.to_tensor().norm() - psi.to_tensor().norm()) < tol


@pytest.mark.parametrize("boundary, dims", [("obc", (2, 3)), ("infinite", (2, 2))])
@pytest.mark.parametrize("key", [None, "reverse"])
def test_env_ctm_to_nonsymmetric(config_kwargs, boundary, dims, key):
    """
    Measurements with converted EnvCTM agree with the symmetric ones.
    With truncation (D_total=12), some connected corner and edge legs carry different charge sectors
    (for the infinite lattice); to_nonsymmetric fills them in, so that dense legs match.
    """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    psi = _random_peps(ops, dims=dims, boundary=boundary)
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 12}, max_sweeps=3)

    env_ns = env.to_nonsymmetric(key=key)
    dops = _dense_ops(ops, key=key)
    site = psi.sites()[0]
    assert env_ns[site].t.config.sym.SYM_ID == 'dense'
    assert env_ns[site].t.get_legs(axes=1).is_fused()  # ket and bra legs stay hard-fused

    sz0, sz1 = env.measure_1site(ops.sz()), env_ns.measure_1site(dops['sz'])
    assert all(abs(sz0[s] - sz1[s]) < tol for s in psi.sites())
    nn0, nn1 = env.measure_nn(ops.sp(), ops.sm()), env_ns.measure_nn(dops['sp'], dops['sm'])
    assert all(abs(nn0[bd] - nn1[bd]) < tol for bd in nn0)


def test_env_ctm_to_nonsymmetric_checkerboard(config_kwargs):
    """ Lattice with a pattern, where several sites share the same tensors. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    ops.config.backend.random_seed(seed=0)
    g = fpeps.CheckerboardLattice()
    v = yastn.Leg(ops.config, s=1, t=(-1, 0, 1), D=(1, 2, 1))
    psi = fpeps.Peps(g, tensors={site: yastn.rand(config=ops.config, legs=[v.conj(), v, v, v.conj(), ops.space()], n=0)
                                 for site in g.sites()})
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 12}, max_sweeps=3)
    env_ns = env.to_nonsymmetric()
    dops = _dense_ops(ops)
    nn0, nn1 = env.measure_nn(ops.sp(), ops.sm()), env_ns.measure_nn(dops['sp'], dops['sm'])
    assert all(abs(nn0[bd] - nn1[bd]) < tol for bd in nn0)


def test_env_ctm_to_nonsymmetric_bra(config_kwargs):
    """ EnvCTM of an overlap <bra|ket> with different bra and ket. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    ket = _random_peps(ops, dims=(2, 2), boundary='obc', seed=0)
    bra = _random_peps(ops, dims=(2, 2), boundary='obc', seed=1)
    env = fpeps.EnvCTM(ket, init='eye', bra=bra)
    env.ctmrg_(opts_svd={'D_total': 12}, max_sweeps=3)
    env_ns = env.to_nonsymmetric()
    assert not env_ns.psi.bra_is_ket
    dops = _dense_ops(ops)
    sz0, sz1 = env.measure_1site(ops.sz()), env_ns.measure_1site(dops['sz'])
    assert all(abs(sz0[s] - sz1[s]) < tol for s in ket.sites())


def test_env_ctm_to_nonsymmetric_purification(config_kwargs):
    """ A hard-fused physical leg (system and ancilla) keeps its fusion. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    g = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
    psi = fpeps.product_peps(g, ops.I())
    for bond in g.bonds():
        psi.apply_gate_(fpeps.gates.gate_nn_Heisenberg(1.0, 0.2, ops.I(), ops.sz(), ops.sp(), ops.sm(), bond))
    site = psi.sites()[0]
    assert psi[site].get_legs(axes=4).is_fused()
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 16}, max_sweeps=3)

    env_ns = env.to_nonsymmetric()
    assert env_ns.psi.ket[site].get_legs(axes=4).is_fused()
    dops = _dense_ops(ops)
    nn0, nn1 = env.measure_nn(ops.sz(), ops.sz()), env_ns.measure_nn(dops['sz'], dops['sz'])
    assert all(abs(nn0[bd] - nn1[bd]) < tol for bd in nn0)


def test_env_boundary_mps_to_nonsymmetric(config_kwargs):
    """ Measurements with converted EnvBoundaryMPS agree with the symmetric ones. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    psi = _random_peps(ops, dims=(2, 3), boundary='obc')
    env = fpeps.EnvBoundaryMPS(psi, opts_svd={'D_total': 16}, setup='lr')
    env_ns = env.to_nonsymmetric()
    dops = _dense_ops(ops)
    sz0, sz1 = env.measure_1site(ops.sz()), env_ns.measure_1site(dops['sz'])
    assert all(abs(sz0[s] - sz1[s]) < tol for s in psi.sites())


@pytest.mark.parametrize("boundary", ["obc", "cylinder"])
def test_transfer_mpo_to_nonsymmetric(config_kwargs, boundary):
    """ Transfer MPO, where sites are DoublePepsTensor (MpoPBC for a cylinder), is converted site by site. """
    ops = yastn.operators.Spin12(sym='U1', **config_kwargs)
    psi = _random_peps(ops, dims=(3, 2), boundary=boundary)
    op = psi.transfer_mpo(n=1, dirn='v')
    assert isinstance(op, mps.MpoPBC) == (boundary == "cylinder")

    op_ns = op.to_nonsymmetric()
    assert type(op_ns) is type(op)
    for n in op.sweep():
        assert isinstance(op_ns[n], fpeps.DoublePepsTensor)
        assert op_ns[n].config.sym.SYM_ID == 'dense'
        assert abs(op_ns[n].fuse_layers().norm() - op[n].fuse_layers().norm()) < tol
    assert op[0].config.sym.SYM_ID == 'U1'  # the original is not modified


def test_to_nonsymmetric_fermionic(config_kwargs):
    """ Swap gates act trivially on non-symmetric tensors; fermionic structures raise an error. """
    ops = yastn.operators.SpinlessFermions(sym='U1', **config_kwargs)
    g = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
    psi = fpeps.product_peps(g, ops.I())
    env = fpeps.EnvCTM(psi, init='eye')
    for obj in (psi, env):
        with pytest.raises(yastn.YastnError,
                           match="to_nonsymmetric\\(\\) is not supported for fermionic config"):
            obj.to_nonsymmetric()


if __name__ == '__main__':
    pytest.main([__file__, "-vs", "--durations=0"])
