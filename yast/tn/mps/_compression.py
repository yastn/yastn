""" Algorithm for variational optimization of mps to match the target state."""
from ._env import Env2, Env3
from ... import initialize, tensor, YastError


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


def zipper(a, b, opts=None):
    "Apply mpo a on mps/mpo b, performing svd compression during the sweep."

    psi = b.clone()
    psi.canonize_sweep(to='last')

    if psi.N != a.N:
        raise YastError('MPS: a and b must have equal number of sites.')

    la, lpsi = a.virtual_leg('last'), psi.virtual_leg('last')

    tmp = initialize.ones(b.config, legs=[lpsi.conj(), la.conj(), lpsi, la])
    tmp = tmp.fuse_legs(axes=(0, 1, (2, 3))).drop_leg_history(axis=2)

    for n in psi.sweep(to='first'):
        tmp = tensor.tensordot(psi[n], tmp, axes=(2, 0))
        if psi.nr_phys == 2:
            tmp = tmp.fuse_legs(axes=(0, 1, 3, (4, 2)))
        tmp = a[n]._attach_23(tmp)

        U, S, V = tensor.svd(tmp, axes=((0, 1), (3, 2)), sU=1)

        mask = tensor.truncation_mask(S, **opts)
        U, C, V = mask.apply_mask(U, S, V, axis=(2, 0, 0))

        psi[n] = V if psi.nr_phys == 1 else V.unfuse_legs(axes=2)
        tmp = U @ C

    tmp = tmp.fuse_legs(axes=((0, 1), 2)).drop_leg_history(axis=0)
    psi[psi.first] = tmp @ psi[psi.first]
    return psi
