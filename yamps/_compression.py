""" Algorithm for variational optimization of mps to match the target state."""
from ._env import Env2, Env3


def variational_sweep_1site(psi, psi_target, env=None, op=None):
    r"""
    Perform a sweep updating psi to maximize the overlap with the target state.

    Operator in a form of mpo can be provided. In that case maximize overlap with op * psi_target.

    Assume input psi is canonical towards first site.
    Sweep consists of iterative updates from last site to first and back to the first one.

    Parameters
    ----------
    psi: Mps
        Initial guess. should be cannonical toward the first site.

    psi_target: Mps
        Target state.

    env: Env2 or Env3
        Environments of the overlap <psi|psi_target> or <psi |op|psi_target> if op is given.
        If None, it is calculated befor the sweep.

    op: Mps
        Mpo acting on psi_target

    Returns
    -------
    env: Env2 or Env3
        environments which can be used during next sweep, or to calculated updated overlap
    """

    if env is None:
        env = Env2(bra=psi, ket=psi_target) if op is None else Env3(bra=psi, op=op, ket=psi_target)
        env.setup(to='first')

    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            psi.remove_central()
            psi.A[n] = env.project_ket_on_bra(n)
            psi.orthogonalize_site(n, to=to)
            env.clear_site(n)
            env.update_env(n, to=to)

    return env
