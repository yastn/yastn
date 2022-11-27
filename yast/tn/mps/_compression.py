""" Algorithm for variational optimization of mps to match the target state."""
from typing import NamedTuple
import logging
from ._env import Env2, Env3
from ... import initialize, tensor, YastError

logger = logging.Logger('compression')

class variational_out(NamedTuple):
    sweeps : int = 0
    overlap : float = None
    doverlap : float = None
    max_dSchmidt : float = None
    max_discarded_weight : float = None


def variational_(psi, op_or_ket, ket_or_none=None, method='1site',
                overlap_tol=None, Schmidt_tol=None, max_sweeps=1,
                iterator_step=None, opts_svd=None):
    r"""
    Perform DMRG sweeps until convergence, starting from MPS :code:`psi`.

    The outer loop sweeps over MPS updating sites from the first site to last and back.
    Convergence can be controlled based on energy and/or Schmidt values (which is a more sensitive measure).
    The DMRG algorithm sweeps through the lattice at most :code:`max_sweeps` times
    or until all convergence measures with provided tolerance change by less then the tolerance.

    Outputs generator if :code:`iterator_step` is given.
    It allows inspecting :code:`psi` outside of :code:`dmrg_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yamps.MpsMpo
        Initial state. It is updated during execution.
        It is first canonized to to the first site, if not provided in such a form.
        State resulting from :code:`dmrg_` is canonized to the first site.

    H: yamps.MpsMpo
        MPO to minimize against.

    project: list(yamps.MpsMpo)
        Optimizes MPS in the subspace orthogonal to MPS's in the list.

    method: str
        Which DMRG variant to use from :code:`'1site'`, :code:`'2site'`

    energy_tol: float
        Convergence tolerance for the change of energy in a single sweep.
        By default is None, in which case energy convergence is not checked.

    energy_tol: float
        Convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is None, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`dmrg_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`dmrg_` sweeps are performed immidiatly.

    opts_eigs: dict
        options passed to :meth:`yast.eigs`.
        If None, use default {'hermitian': True, 'ncv': 3, 'which': 'SR'}

    opts_svd: dict
        Options passed to :meth:`yast.svd` used to truncate virtual spaces in :code:`method='2site'`.
        If None, use default {'tol': 1e-14}

    Returns
    -------
    out: DMRGout(NamedTuple)
        Includes fields:
        :code:`sweeps` number of performed dmrg sweeps.
        :code:`energy` energy after the last sweep.
        :code:`denergy` absolut value of energy change in the last sweep.
        :code:`max_dSchmidt` norm of Schmidt values change on the worst cut in the last sweep
        :code:`max_discarded_weight` norm of discarded_weights on the worst cut in '2site' procedure.
    """
    tmp = _variational_(psi, op_or_ket, ket_or_none, method,
                        overlap_tol, Schmidt_tol, max_sweeps,
                        iterator_step, opts_svd)
    return tmp if iterator_step else next(tmp)


def _variational_(psi, op_or_ket, ket_or_none, method,
                overlap_tol, Schmidt_tol, max_sweeps,
                iterator_step, opts_svd):
    """ Generator for variational_(). """

    if not psi.is_canonical(to='first'):
        psi.canonize_sweep(to='first')

    env = Env2(bra=psi, ket=op_or_ket) if ket_or_none is None else \
        Env3(bra=psi, op=op_or_ket, ket=ket_or_none)
    env.setup(to='first')

    overlap_old = env.measure()

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastError('DMRG: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
    max_dS, max_dw = None, None
    Schmidt = None if Schmidt_tol is None else {}

    if overlap_tol is not None and not overlap_tol > 0:
        raise YastError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', '2site'):
        raise YastError('DMRG: dmrg method %s not recognized.' % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _variational_1site_sweep_(env, Schmidt=Schmidt)
        else: # method == '2site':
            max_dw = _variational_2site_sweep_(env, opts_svd=opts_svd, Schmidt=Schmidt)

        overlap = env.measure()
        doverlap, overlap_old = overlap_old - overlap, overlap
        converged = []

        if overlap_tol is not None:
            converged.append(abs(doverlap) < overlap_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm() for k in Schmidt.keys())
            Schmidt_old = Schmidt
            converged.append(max_dS < Schmidt_tol)

        logger.info('Sweep = %03d  overlap = %0.14f  doverlap = %0.4f  dSchmidt = %0.4f', sweep, overlap, doverlap, max_dS)

        if len(converged) > 0 and all(converged):
            break
        if iterator_step and sweep % iterator_step == 0 and sweep < max_sweeps:
            yield variational_out(sweep, overlap, doverlap, max_dS, max_dw)
    yield variational_out(sweep, overlap, doverlap, max_dS, max_dw)


def _variational_1site_sweep_(env, Schmidt):
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

    psi = env.bra
    for to in ('last', 'first'):
        for n in psi.sweep(to=to):
            psi.remove_central()
            psi.A[n] = env.project_ket_on_bra(n)
            psi.orthogonalize_site(n, to=to)
            env.clear_site(n)
            env.update_env(n, to=to)


def _variational_2site_sweep_(env):
    pass


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
