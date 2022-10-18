""" Algorithm for variational optimization of mps to match the target state."""
from ._env import Env2, Env3
from. _mps import YampsError
import yast

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


def multiply_svd(a, b, opts=None):
    "Apply mpo a on mps/mpo b, performing svd compression during the sweep."
    psi = b.clone()
    psi.canonize_sweep(to='last')

    if b.N != a.N:
        raise YampsError('a and b must have equal number of sites.')
    if a.pC is not None or b.pC is not None:
        raise YampsError('Absorb central sites of mps-s before calling multiply.')

    axes_fuse = (3, 0, (2, 4), 1) if b.nr_phys == 1 else (3, 0, (2, 4), 5, 1)
    tmp = yast.tensordot(a[a.last], b[b.last], axes=(3, 1))
    tmp = tmp.fuse_legs(axes_fuse).drop_leg_history(axis=2)
    if b.nr_phys == 2:
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 3), 4))

    for n in psi.sweep(to='first'):
        U, S, V = yast.svd(tmp, axes=((0, 1), (3, 2)), sU=-1)

        mask = yast.linalg.truncation_mask(S, **opts)
        U, C, V = mask.apply_mask(U, S, V, axis=(2, 0, 0))

        psi.A[n] = V if b.nr_phys == 1 else V.unfuse_legs(axes=2)
        UC = U @ C

        if n > psi.first:
            tmp = yast.tensordot(b[n-1], UC, axes=(2, 0))
            if b.nr_phys == 2:
                tmp = tmp.fuse_legs(axes=(0, 1, 3, (4, 2)))
            tmp = a[n-1]._attach_23(tmp)
    UC = UC.fuse_legs(axes=((0, 1), 2)).drop_leg_history(axis=0)
    psi.A[psi.first] = UC @ psi.A[psi.first]
    return psi
