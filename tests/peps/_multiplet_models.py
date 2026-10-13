"""Model Hamiltonians built from public scalar and multiplet operators."""
from yastn.tn import mps


def hubbard_hamiltonian(ops, t, U, mu):
    c, cp = ops.c_spinor(), ops.cp_spinor()
    local = U * ops.d() - mu * ops.n_total()
    return mps.generate_mpo(ops.I(), [
        mps.Hterm(-t, (0, 1), (cp, c)),
        mps.Hterm(-t, (1, 0), (cp, c)),
        mps.Hterm(1, (0,), (local,)),
        mps.Hterm(1, (1,), (local,)),
    ], N=2)


def tj_hamiltonian(ops, J, t, mu0, mu1):
    c, cp, S, n = ops.c_spinor(), ops.cp_spinor(), ops.S(), ops.n_total()
    return mps.generate_mpo(ops.I(), [
        mps.Hterm(-t, (0, 1), (cp, c)),
        mps.Hterm(-t, (1, 0), (cp, c)),
        mps.Hterm(J, (0, 1), (S, S)),
        mps.Hterm(-J / 4, (0, 1), (n, n)),
        mps.Hterm(-mu0, (0,), (n,)),
        mps.Hterm(-mu1, (1,), (n,)),
    ], N=2)
