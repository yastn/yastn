""" Algorithm for variational optimization of mps to match the target state."""
from ._env import Env2, Env3

def variational_sweep_1site(psi, psi_target, env=None, op=None):
    r"""
    Using :code:`verions='1site'` DMRG, an MPS :code:`psi` with fixed 
    virtual spaces is variationally optimized to maximize overlap 
    :math:`\langle \psi | \psi_{\textrm{target}}\rangle` with 
    the target MPS :code:`psi_target`.

    The principal use of this algorithm is an (approximate) compression of large MPS 
    into MPS with smaller virtual dimension/spaces. 

    Operator in a form of MPO can be provided in which case algorithm maximizes 
    overlap :math:`\langle \psi | O |\psi_{target}\rangle`.

    It is assumed that the initial MPS :code:`psi` is in the right canonical form. 
    The outer loop sweeps over MPS :code:`psi` updating sites from the first site to last and back. 

    Parameters
    ----------
    psi: yamps.MpsMpo
        initial MPS in right canonical form.

    psi_target: yamps.MpsMpo
        Target MPS.

    env: Env2 or Env3
        optional environment of tensor network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle` from the previous run.

    op: yamps.MpsMpo
        operator acting on :math:`|\psi_{\textrm{target}}\rangle`.

    Returns
    -------
    env: yamps.Env2 or yamps.Env3
        Environment of the network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle`.
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
