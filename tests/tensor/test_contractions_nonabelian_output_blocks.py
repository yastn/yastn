"""Non-Abelian tensordot must not assume its output classes are all in struct_c.

Background: the no-fusion metadata builder computes an index from the produced
output classes to the output blocks. That lookup assumes every produced class is
present in the output structure, which holds for the Abelian symmetries but not
for the non-Abelian ones: there a contraction may legitimately produce an output
class that ``struct_c`` does not carry, and the branch that consumes the metadata
matches classes by charge and fusion channel and drops absent ones. The lookup is
unused on that branch, so it must not run there at all.

The tests below compare the ``no_fusion`` metadata path against the
``fuse_contracted`` path for SU(2), SU(2xU1) and U(1xU1xZ2), covering contracted
axes at the boundary and in the interior, and pin the lookup to the Abelian
branch only.
"""
import numpy as np
import pytest

import yastn
from yastn.sym import sym_SU2, sym_SU2xU1, sym_U1xU1xZ2

SYMMETRIES = [sym_SU2, sym_SU2xU1, sym_U1xU1xZ2]
SYM_IDS = ['SU2', 'SU2xU1', 'U1xU1xZ2']


def _charges(sym):
    if sym is sym_U1xU1xZ2:
        return ((0, 0, 0), (1, 0, 1), (0, 1, 1))
    if sym is sym_SU2xU1:
        return ((0, 0), (1, 1), (2, 0))
    return ((0,), (1,), (2,))


def _config(sym, policy):
    kwargs = {'sym': sym, 'backend': 'np', 'default_dtype': 'float64',
              'tensordot_policy': policy}
    if sym is sym_U1xU1xZ2:
        kwargs['fermionic'] = (False, False, True)
    return yastn.make_config(**kwargs)


def _legs(cfg, sym, rank):
    leg = yastn.Leg(cfg, s=1, t=_charges(sym), D=(2,) * len(_charges(sym)))
    return tuple(leg if i % 2 == 0 else leg.conj() for i in range(rank))


def _contract(sym, rank, contracted, policy):
    cfg = _config(sym, policy)
    legs = _legs(cfg, sym, rank)
    cfg.backend.random_seed(seed=7)
    a = yastn.rand(cfg, legs=legs)
    cfg.backend.random_seed(seed=11)
    b = yastn.rand(cfg, legs=tuple(legs[i].conj() for i in contracted))
    return yastn.tensordot(a, b, axes=(contracted, tuple(range(len(contracted)))))


@pytest.mark.parametrize('sym, sym_id', list(zip(SYMMETRIES, SYM_IDS)), ids=SYM_IDS)
@pytest.mark.parametrize('rank,contracted', [(2, (0,)), (4, (0,)), (4, (1,)), (4, (1, 3))],
                         ids=['rank2-first', 'rank4-first', 'rank4-interior', 'rank4-two-axes'])
def test_no_fusion_matches_fused_tensordot(sym, sym_id, rank, contracted):
    """The no_fusion metadata path agrees with the fuse_contracted path."""
    c_nf = _contract(sym, rank, contracted, 'no_fusion')
    c_fused = _contract(sym, rank, contracted, 'fuse_contracted')
    assert c_nf.shape == c_fused.shape
    assert np.allclose(c_nf.to_numpy(), c_fused.to_numpy(), atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Regression for the missing-output-class case. A finite (OBC) purification
# PEPS with the t-J local space produces an environment whose output structure
# legitimately lacks one produced class on some bonds; the lookup must not run
# there. This needs a converged CTM at D=20, hence --long_tests.
# ---------------------------------------------------------------------------
LONG = pytest.mark.skipif("not config.getoption('long_tests')",
                          reason="long duration test; pass --long_tests to run it")


@LONG
def test_su2xu1_ctm_measurement_with_absent_output_classes():
    """SU2xU1 CTM measurement no longer aborts on a missing output class."""
    from yastn.tn import fpeps, mps

    cfg = yastn.make_config(sym='SU2xU1', backend='np', fermionic=(False, True),
                            default_dtype='float64')
    ops = yastn.operators.SpinfulFermions_tJ(sym='SU2xU1', backend='np')
    t, U, mu = 1.0, 4.0, 0.3
    J = 4 * t**2 / U
    n, cp, c, S = ops.n_total(), ops.cp_spinor(), ops.c_spinor(), ops.S()
    terms = [mps.Hterm(-t, (0, 1), (cp, c)), mps.Hterm(-t, (1, 0), (cp, c)),
             mps.Hterm(J, (0, 1), (S, S)), mps.Hterm(-J / 4, (0, 1), (n, n)),
             mps.Hterm(-mu / 2, (0,), (n,)), mps.Hterm(-mu / 2, (1,), (n,))]
    gate = fpeps.gates.gate_nn_exp(0.05 / 4, ops.I(), mps.generate_mpo(ops.I(), terms, N=2))
    net = fpeps.SquareLattice(dims=(2, 2), boundary='obc')
    psi = fpeps.product_peps(net, ops.I() / np.sqrt(3.))
    envntu = fpeps.EnvNTU(psi, which='NN')
    gates = fpeps.gates.distribute(net, gates_nn=gate, symmetrize=True)
    fpeps.evolution_step_(envntu, gates, opts_svd={'D_total': 20, 'tol': 1e-12},
                          max_iter=4, initialization='EAT')
    env = fpeps.EnvCTM(psi, init='eye')
    env.ctmrg_(opts_svd={'D_total': 24, 'tol': 1e-8, 'truncate_multiplets': True},
               max_sweeps=6, corner_tol=None)
    bonds = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 0), (0, 1)), ((1, 0), (1, 1))]
    for bond in bonds:
        for a, b in ((S, S), (cp, c), (n, n)):
            assert np.isfinite(np.real(env.measure_nn(a, b, bond=bond)))
