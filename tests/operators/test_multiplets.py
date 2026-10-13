"""Irreducible operators are ordinary invariant three-leg tensors."""
import pytest
import yastn
from yastn.operators._multiplets import _is_multiplet, _multiplet_pair


@pytest.mark.parametrize('sym', ['SU2', 'SU2xU1'])
@pytest.mark.parametrize('factory', [yastn.operators.SpinfulFermions, yastn.operators.SpinfulFermions_tJ])
def test_multiplet_operators(config_kwargs, sym, factory):
    ops = factory(sym=sym, **config_kwargs)
    S, cp, c = ops.S(), ops.cp_spinor(), ops.c_spinor()
    charges = ((2,), (1,), (1,)) if sym == 'SU2' else ((2, 0), (1, 1), (1, 1))
    assert not _is_multiplet(ops.I())
    for op, charge, signature in zip((S, cp, c), charges, (-1, -1, 1)):
        assert type(op) is yastn.Tensor and _is_multiplet(op)
        assert op.ndim == 3 and op.n == op.config.sym.zero()
        assert op.s == (1, -1, signature)
        leg = op.get_legs(2)
        assert leg.t == (charge,) and leg.D == (1,)
        for other in (op.copy(), op.clone(), op.detach(), op.to(), 2 * op / 2,
                      yastn.from_dict(op.to_dict()), yastn.from_dict(op.to_dict(resolve_ops=True))):
            assert (other - op).norm() < 2e-12
        # Multiplicity one is not an Abelian scalar: closing the identity on
        # this irrep sums over all magnetic states, giving 2j+1, not one.
        identity = yastn.eye(op.config, legs=(leg.conj(), leg), isdiag=False)
        assert identity.trace(axes=(0, 1)).to_number() == pytest.approx(charge[0] + 1)
    A, B = _multiplet_pair(cp, c)
    assert (yastn.tensordot(A, B, axes=((1, 2), (0, 2))) - ops.n_total()).norm() < 3e-12
    assert (cp.conj().transpose((1, 0, 2)) - c).norm() < 2e-12
    assert (c.conj().transpose((1, 0, 2)) - cp).norm() < 2e-12


@pytest.mark.parametrize('factory', [yastn.operators.Spin12, yastn.operators.Spin1])
def test_spin_multiplet(config_kwargs, factory):
    ops = factory(sym='SU2', **config_kwargs)
    S = ops.S()
    assert type(S) is yastn.Tensor and S.ndim == 3 and S.n == (0,)
    A, B = _multiplet_pair(S, S)
    casimir = 0.75 if factory is yastn.operators.Spin12 else 2
    assert (yastn.tensordot(A, B, axes=((1, 2), (0, 2))) - casimir * ops.I()).norm() < 3e-12
