""" Algorithm for variational optimization of mps to match the target state."""
from typing import NamedTuple
import logging
from ._env import Env2, Env3
from ._mps import MpsMpo
from ... import initialize, tensor, YastnError

logger = logging.Logger('compression')

class compression_out(NamedTuple):
    sweeps : int = 0
    overlap : float = None
    doverlap : float = None
    max_dSchmidt : float = None
    max_discarded_weight : float = None


def compression_(psi, target, method='1site',
                overlap_tol=None, Schmidt_tol=None, max_sweeps=1,
                iterator_step=None, opts_svd=None, normalize=True):
    r"""
    Perform variational optimization sweeps until convergence to best approximate the target, starting from MPS/MPO :code:`psi`.

    The outer loop sweeps over ``psi`` updating sites from the first site to last and back.
    Convergence can be controlled based on overlap and/or Schmidt values (which is a more sensitive measure).
    The algorithm performs at most :code:`max_sweeps`. If tolerance measures are provided, it terminates earlier 
    if the convergence criteria are satisfied: change in overlap or Schmidt values is less then the provided tolerance during a single sweep.

    Works for 

        * optimization against provided MPS: ``target`` is ``Mps`` or list ``[Mps,]``
        * against MPO acting on MPS: ``target`` is a list ``[Mpo, Mps]``.
        * against MPO (replacing all Mps's above with Mpo's), i.e., ``[Mpo,]`` or ``[Mpo, Mpo]``
        * sum of MPS's: target is ``[[Mps],[Mps],...]``
        * sum of MPO's acting on MPS's: target is ``[[Mpo,Mps], [Mpo,Mps], ...]``
 
    Outputs generator if :code:`iterator_step` is given.
    Generator allows inspecting :code:`psi` outside of :code:`compression_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yastn.tn.mps.MpsMpo
        Initial state. It is updated during execution.
        It is first canonized to the first site, if not provided in such a form.
        State resulting from :code:`compression_` is canonized to the first site.

    target: yamps.MpsMpo or list(yamps.MpsMpo)
        Defines target state. Can be an Mps (target = Mps or (Mps,)),
        or Mpo acting on Mps (target = (Mpo, Mps)).

    method: str
        Which optimization variant to use from :code:`'1site'`, :code:`'2site'`

    overlap_tol: float
        Convergence tolerance for the change of overlap in a single sweep.
        By default is None, in which case overlap convergence is not checked.

    Schmidt_tol: float
        Convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is None, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`compression_` returns a generator that would yield output after every iterator_step sweeps.
        Default is None, in which case  :code:`compression_` sweeps are performed immidiatly.

    opts_svd: dict
        Options passed to :meth:`yastn.svd` used to truncate virtual spaces in :code:`method='2site'`.
        If None, use default {'tol': 1e-14}

    Returns
    -------
    compression_out(NamedTuple)
        NamedTuple with fields

            * :code:`sweeps` number of performed sweeps.
            * :code:`overlap` overlap after the last sweep.
            * :code:`doverlap` absolute value of overlap change in the last sweep.
            * :code:`max_dSchmidt` norm of Schmidt values change on the worst cut in the last sweep.
            * :code:`max_discarded_weight` norm of discarded_weights on the worst cut in '2site' procedure.
    """
    tmp = _compression_(psi, target, method,
                        overlap_tol, Schmidt_tol, max_sweeps,
                        iterator_step, opts_svd, normalize)
    return tmp if iterator_step else next(tmp)


def _compression_(psi, target, method,
                overlap_tol, Schmidt_tol, max_sweeps,
                iterator_step, opts_svd, normalize):
    """ Generator for compression_(). """

    if not psi.is_canonical(to='first'):
        psi.canonize_(to='first')

    if isinstance(target, MpsMpo):
        env = Env2(bra=psi, ket=target)
    elif len(target) == 1:
        env = Env2(bra=psi, ket=target[0])
    else:
        env = Env3(bra=psi, op=target[0], ket=target[1])

    env.setup(to='first')

    overlap_old = env.measure()

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastnError('DMRG: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
    max_dS, max_dw = None, None
    Schmidt = None if Schmidt_tol is None else {}

    if overlap_tol is not None and not overlap_tol > 0:
        raise YastnError('DMRG: energy_tol has to be positive or None.')

    if method not in ('1site', '2site'):
        raise YastnError('DMRG: dmrg method %s not recognized.' % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _compression_1site_sweep_(env, Schmidt=Schmidt)
        else: # method == '2site':
            max_dw = _compression_2site_sweep_(env, opts_svd=opts_svd, Schmidt=Schmidt)

        psi.factor = 1
        overlap = env.measure()
        doverlap, overlap_old = overlap_old - overlap, overlap
        converged = []

        if not normalize:
            psi.factor = overlap

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
            yield compression_out(sweep, overlap, doverlap, max_dS, max_dw)
    yield compression_out(sweep, overlap, doverlap, max_dS, max_dw)


def _compression_1site_sweep_(env, Schmidt=None):
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
    psi: yastn.tn.mps.MpsMpo
        initial MPS in right canonical form.

    psi_target: yastn.tn.mps.MpsMpo
        Target MPS.

    env: Env2 or Env3
        optional environment of tensor network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle` from the previous run.

    op: yastn.tn.mps.MpsMpo
        operator acting on :math:`|\psi_{\textrm{target}}\rangle`.

    Returns
    -------
    env: yastn.tn.mps.Env2 or yastn.tn.mps.Env3
        Environment of the network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle`.
    """
    bra, ket = env.bra, env.ket
    for to in ('last', 'first'):
        for n in bra.sweep(to=to):
            bra.remove_central()
            bra.A[n] = env.Heff1(ket[n], n)
            bra.orthogonalize_site(n, to=to, normalize=True)
            if Schmidt is not None and to == 'first' and n != bra.first:
                _, S, _ = bra[bra.pC].svd(sU=1)
                Schmidt[bra.pC] = S
            env.clear_site(n)
            env.update_env(n, to=to)
    bra.absorb_central(to='first')
    env.update_env(n, to=to)



def _compression_2site_sweep_(env, opts_svd=None, Schmidt=None):
    """ variational update on 2 sites """
    if opts_svd is None:
        opts_svd = {'tol': 1e-14}
    max_disc_weight = -1.
    bra, ket = env.bra, env.ket
    for to, dn in (('last', 0), ('first', 1)):
        for n in bra.sweep(to=to, dl=1):
            bd = (n, n + 1)
            AA = ket.merge_two_sites(bd)
            AA = env.Heff2(AA, bd)
            _disc_weight_bd = bra.unmerge_two_sites(AA, bd, opts_svd)
            max_disc_weight = max(max_disc_weight, _disc_weight_bd)
            if Schmidt is not None and to == 'first':
                Schmidt[bra.pC] = bra[bra.pC]
            bra.absorb_central(to=to)
            env.clear_site(n, n + 1)
            env.update_env(n + dn, to=to)
    bra[bra.first] = bra[bra.first] / bra[bra.first].norm()
    env.update_env(bra.first, to='first')
    return max_disc_weight


def zipper(a, b, opts=None):
    "Apply MPO a on MPS/MPS b, performing svd compression during the sweep."

    psi = b.shallow_copy()
    psi.canonize_(to='last')

    if psi.N != a.N:
        raise YastnError('MPS: a and b must have equal number of sites.')

    la, lpsi = a.virtual_leg('last'), psi.virtual_leg('last')

    tmp = initialize.ones(b.config, legs=[lpsi.conj(), la.conj(), lpsi, la])
    tmp = tmp.fuse_legs(axes=(0, 1, (2, 3))).drop_leg_history(axes=2)

    for n in psi.sweep(to='first'):
        tmp = tensor.tensordot(psi[n], tmp, axes=(2, 0))
        
        if psi.nr_phys == 2:
            tmp = tmp.fuse_legs(axes=(0, 1, 3, (4, 2)))
        tmp = a[n]._attach_23(tmp)

        U, S, V = tensor.svd(tmp, axes=((0, 1), (3, 2)), sU=1)

        mask = tensor.truncation_mask(S, **opts)
        U, C, V = mask.apply_mask(U, S, V, axes=(2, 0, 0))

        psi[n] = V if psi.nr_phys == 1 else V.unfuse_legs(axes=2)
        tmp = U @ C

    tmp = tmp.fuse_legs(axes=((0, 1), 2)).drop_leg_history(axes=0)
    ntmp = tmp.norm()
    psi[psi.first] = (tmp / ntmp) @ psi[psi.first]
    psi.factor = a.factor * b.factor * ntmp
    return psi