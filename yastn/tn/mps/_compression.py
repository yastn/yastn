# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Algorithm for variational optimization of Mps to match the target state."""
from __future__ import annotations
import logging
from typing import NamedTuple

from ._measure import Env
from ._mps_obc import MpsMpoOBC, MpoPBC
from ...initialize import ones, eye
from ...tensor import YastnError

logger = logging.Logger('compression')


class compression_out(NamedTuple):
    sweeps: int = 0
    overlap: float = None
    doverlap: float = None
    max_dSchmidt: float = None
    max_discarded_weight: float = None


def compression_(psi, target, method='1site',
                overlap_tol=None, Schmidt_tol=None, max_sweeps=1,
                iterator_step=None, opts_svd=None, normalize=True):
    r"""
    Perform variational optimization sweeps until convergence to best approximate the target, starting from MPS/MPO :code:`psi`.

    The outer loop sweeps over ``psi`` updating sites from the first site to the last and back.
    Convergence can be controlled based on overlap and/or Schmidt values (which is a more sensitive measure of convergence).
    The algorithm performs at most :code:`max_sweeps`. If tolerance measures are provided,
    the calculation ends when the convergence criteria are satisfied, e.g., change in overlap or Schmidt values between sweeps
    is less than the provided tolerance.

    Works for

        * optimization against provided MPS: ``target`` is ``MPS`` or list ``[MPS]``
        * against MPO acting on MPS: ``target`` is a list ``[MPO, MPS]``.
        * against sum of MPOs acting on MPS: ``target`` is a list ``[[MPO, MPO, ...], MPS]``.
        * sum of any of the three above: target is ``[[MPS], [MPO, MPS], [[MPO, MPO, ...], MPS], ...]``
        * for ``psi`` being itself an MPO, all MPS's above should be replaced with MPO, e.g., ``[MPO, MPO]``

    Outputs iterator if :code:`iterator_step` is given, which allows
    inspecting :code:`psi` outside of :code:`compression_` function after every :code:`iterator_step` sweeps.

    Parameters
    ----------
    psi: yastn.tn.mps.MpsMpoOBC
        Initial state. It is updated during execution.
        It is first canonized to the first site, if not provided in such a form.
        State resulting from :code:`compression_` is canonized to the first site.

    target: MPS or MPO
        Defines target state. The target can be:

        * an MPS, e.g. target = MPS or [MPS],
        * an MPO acting on MPS (target = [MPO, MPS]),
        * sum of MPOs acting on MPS, e.g, [[MPO, MPO], MPS],
        * or the sum of the above, e.g., [[MPS], [MPO, MPS], [[MPO, MPO], MPS]].
        * If ``psi`` and ``target`` are MPOs, the MPS in above list is replaced by MPO.

    method: str
        Which optimization variant to use from `'1site'`, `'2site'`

    overlap_tol: float
        Convergence tolerance for the change of relative overlap in a single sweep.
        By default is `None`, in which case overlap convergence is not checked.

    Schmidt_tol: float
        Convergence tolerance for the change of Schmidt values on the worst cut/bond in a single sweep.
        By default is `None`, in which case Schmidt values convergence is not checked.

    max_sweeps: int
        Maximal number of sweeps.

    iterator_step: int
        If int, :code:`compression_` returns a generator that would yield output after every iterator_step sweeps.
        The default is None, in which case  :code:`compression_` sweeps are performed immediately.

    opts_svd: dict
        Options passed to :meth:`yastn.linalg.svd` used to truncate virtual spaces in :code:`method='2site'`.

    Returns
    -------
    compression_out(NamedTuple)
        NamedTuple with fields

            * :code:`sweeps` number of performed sweeps.
            * :code:`overlap` overlap after the last sweep.
            * :code:`doverlap` absolute value of relative overlap change in the last sweep.
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

    env = Env(psi, target)
    env.setup_(to='first')

    overlap_old = env.measure()

    if Schmidt_tol is not None:
        if not Schmidt_tol > 0:
            raise YastnError('Compression: Schmidt_tol has to be positive or None.')
        Schmidt_old = psi.get_Schmidt_values()
        Schmidt_old = {(n-1, n): sv for n, sv in enumerate(Schmidt_old)}
    max_dS, max_dw = None, None
    Schmidt = None if Schmidt_tol is None else {}

    if overlap_tol is not None and not overlap_tol > 0:
        raise YastnError('Compression: overlap_tol has to be positive or None.')

    if opts_svd is None and method == '2site':
        raise YastnError("Compression: provide opts_svd for %s method." % method)

    if method not in ('1site', '2site'):
        raise YastnError('Compression: method %s not recognized.' % method)

    for sweep in range(1, max_sweeps + 1):
        if method == '1site':
            _compression_1site_sweep_(env, Schmidt=Schmidt)
        else: # method == '2site':
            max_dw = _compression_2site_sweep_(env, opts_svd=opts_svd, Schmidt=Schmidt)

        psi.factor = 1
        overlap = env.measure()
        doverlap = (overlap - overlap_old) / overlap if overlap else 0.
        overlap_old = overlap
        converged = []

        if not normalize:
            psi.factor = overlap

        if overlap_tol is not None:
            converged.append(abs(doverlap) < overlap_tol)

        if Schmidt_tol is not None:
            max_dS = max((Schmidt[k] - Schmidt_old[k]).norm() for k in Schmidt.keys())
            Schmidt_old = Schmidt.copy()
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
    The outer loop sweeps over MPS :code:`psi` updating sites from the first site to the last and back.

    Parameters
    ----------
    psi: yastn.tn.mps.MpsMpoOBC
        initial MPS in right canonical form.

    psi_target: yastn.tn.mps.MpsMpoOBC
        Target MPS.

    env: Env2 or Env3
        optional environment of tensor network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle` from the previous run.

    op: yastn.tn.mps.MpsMpoOBC
        operator acting on :math:`|\psi_{\textrm{target}}\rangle`.

    Returns
    -------
    env: yastn.tn.mps.Env2 or yastn.tn.mps.Env3
        Environment of the network :math:`\langle \psi|\psi_{target} \rangle`
        or :math:`\langle \psi|O|\psi_{target} \rangle`.
    """
    bra = env.bra
    for to in ('last', 'first'):
        for n in bra.sweep(to=to):
            bra.remove_central_()
            A = env.project_ket_on_bra_1(n)
            bra.post_1site_(A, n)
            bra.orthogonalize_site_(n, to=to, normalize=True)
            if Schmidt is not None and to == 'first' and n != bra.first:
                Schmidt[bra.pC] = bra[bra.pC].svd(sU=1, compute_uv=False)
            env.clear_site_(n)
            env.update_env_(n, to=to)
    bra.absorb_central_(to='first')
    env.update_env_(n, to=to)


def _compression_2site_sweep_(env, opts_svd=None, Schmidt=None):
    """ variational update on 2 sites """
    max_disc_weight = -1.
    bra = env.bra
    for to, dn in (('last', 0), ('first', 1)):
        for n in bra.sweep(to=to, dl=1):
            bd = (n, n + 1)
            AA = env.project_ket_on_bra_2(bd)
            _disc_weight_bd = bra.post_2site_(AA, bd, opts_svd)
            pCnorm = bra.A[bra.pC].norm()
            if pCnorm:
                bra.A[bra.pC] = bra.A[bra.pC] / pCnorm
            max_disc_weight = max(max_disc_weight, _disc_weight_bd)
            if Schmidt is not None and to == 'first':
                Schmidt[bra.pC] = bra[bra.pC]
            bra.absorb_central_(to=to)
            env.clear_site_(n, n + 1)
            env.update_env_(n + dn, to=to)
    env.update_env_(bra.first, to='first')
    return max_disc_weight


def zipper(a, b, opts_svd=None, normalize=True, return_discarded=False) -> MpsMpoOBC:
    """
    Apply MPO ``a`` on MPS/MPS ``b``, performing SVD compression during the sweep.

    Perform canonization of ``b`` to the last site.
    Next, sweep back, attaching to it elements of ``a`` one at a time,
    truncating the resulting bond dimensions along the way.
    The resulting state is canonized to the first site and normalized to unity.

    Parameters
    ----------
    a, b: yastn.tn.mps.MpsMpoOBC

    opts_svd: dict
        truncation parameters for :meth:`yastn.linalg.truncation_mask`.

    normalize: bool
        Whether to keep track of the norm of the initial state projected on
        the direction of the truncated state; default is ``True``, i.e., sets the norm to :math:`1`.
        The individual tensors at the end of the procedure are in a proper canonical form.

    return_discarded: bool
        Whether to return the approximation discarded weights together with the resulting MPS/MPO.
        The default is ``False``, i.e., returns only MPS/MPO.
        Discarded weight approximates norm of truncated elements normalized by the norm of the untruncated state.
    """
    if a.N != b.N:
        raise YastnError('Zippr: Mpo and Mpo/Mps must have the same number of sites to be multiplied.')

    psi = b.shallow_copy()
    psi.canonize_(to='last', normalize=normalize)
    if not normalize:
        psi.factor = psi.factor * a.factor

    if isinstance(a, MpoPBC):
        if psi.nr_phys != 1:
            raise YastnError("Zipper: Application of MpoPBC on Mpo is currently not supported. Contact developers to add this functionality.")
        psi, discarded = _zipper_MpoPBC(a, psi, opts_svd, normalize)

    if isinstance(a, MpsMpoOBC) and a.nr_phys == 2:
        psi, discarded = _zipper_MpoOBC(a, psi, opts_svd, normalize)

    if return_discarded:
        return psi, discarded
    return psi


def _zipper_MpoOBC(a, psi, opts_svd, normalize) -> MpsMpoOBC:
    """
    Special case of MpoOBC
    """

    la, lpsi = a.virtual_leg('last'), psi.virtual_leg('last')

    tmp = ones(psi.config, legs=[lpsi.conj(), la.conj(), lpsi, la])
    tmp = tmp.fuse_legs(axes=(0, 1, (2, 3))).drop_leg_history(axes=2)

    discarded2_total = 0.
    for n in psi.sweep(to='first'):
        tmp = psi[n].tensordot(tmp, axes=(2, 0))
        if psi.nr_phys == 2:
            tmp = tmp.fuse_legs(axes=(0, 1, 3, (4, 2)))
        tmp = a[n].tensordot(tmp, axes=((2, 3), (2, 1)))

        U, S, V = tmp.svd(axes=((2, 0), (1, 3)), sU=1, nU=False)
        nSold = S.norm()

        mask = S.truncation_mask(**opts_svd)
        nSout = mask.bitwise_not().apply_mask(S, axes=0).norm()
        discarded2_local = (nSout / nSold) ** 2 if nSold else 0.
        discarded2_total = discarded2_total + discarded2_local - discarded2_total * discarded2_local

        U, S, V = mask.apply_mask(U, S, V, axes=(2, 0, 0))
        nS = S.norm()

        psi[n] = V if psi.nr_phys == 1 else V.unfuse_legs(axes=2)
        if nS:
            S = S / nS
        tmp = U @ S
        psi.factor = psi.factor * nS

    tmp = tmp.fuse_legs(axes=((0, 1), 2)).drop_leg_history(axes=0)
    ntmp = tmp.norm()
    if ntmp:
        tmp = tmp / ntmp
    psi[psi.first] = tmp @ psi[psi.first]
    psi.factor = 1 if normalize else psi.factor * ntmp

    return psi, discarded2_total ** 0.5


def _zipper_MpoPBC(a, psi, opts_svd, normalize) -> MpsMpoOBC:
    """
    Special case of MpoPBC
    """
    lmpo, lpsi = a.virtual_leg('last'), psi.virtual_leg('last')

    tmp = eye(psi.config, legs=lmpo, isdiag=False)
    tmp = tmp.tensordot(eye(psi.config, legs=lpsi, isdiag=False), axes=((), ()))
    tmp = tmp.transpose(axes=(3, 0, 1, 2))
    # tmp = tmp.add_leg(axis=0, s=-lpsi.s, t=lpsi.t[0])
    # tmp = tmp.add_leg(axis=3, s=lpsi.s, t=lpsi.t[0])

    connector = eye(psi.config, legs=lmpo, isdiag=False)

    discarded2_total = 0.
    for n in psi.sweep(to='first'):
        tmp = psi[n].tensordot(tmp, axes=(2, 0))
        #if psi.nr_phys == 1:
        tmp = tmp.fuse_legs(axes=((0, 2), 1, 3, 4))
        # else:  # psi.nr_phys == 2:
        #     tmp = tmp.swap_gate(axes=(2, 3))
        #     tmp = tmp.fuse_legs(axes=((0, 3), 1, 4, (5, 2)))
        tmp = a[n].tensordot(tmp, axes=((2, 3), (2, 1)))

        if n > psi.first:
            U, S, V = tmp.svd(axes=((2, 0), (1, 3)), sU=1, nU=False)
            nSold = S.norm()

            mask = S.truncation_mask(**opts_svd)
            nSout = mask.bitwise_not().apply_mask(S, axes=0).norm()
            discarded2_local = (nSout / nSold) ** 2 if nSold else 0.
            discarded2_total = discarded2_total + discarded2_local - discarded2_total * discarded2_local

            U, S, V = mask.apply_mask(U, S, V, axes=(2, 0, 0))
            nS = S.norm()

            psi[n] = V if psi.nr_phys == 1 else V.unfuse_legs(axes=2)
            if nS:
                S = S / nS
            tmp = (U @ S).unfuse_legs(axes=0)

            if a.tol is not None and a.tol > 0:
                Uc, Sc, Vc = tmp.svd_with_truncation(axes=(1, (0, 2, 3)), tol=a.tol)
                tmp = (Sc @ Vc).transpose(axes=(1, 0, 2, 3))
                connector = connector @ Uc

            psi.factor = psi.factor * nS
        else:  # n == first
            tmp = tmp.unfuse_legs(axes=2)
            tmp = connector.tensordot(tmp, axes=(1, 3))
            tmp = tmp.trace(axes=(0, 1))
            ntmp = tmp.norm()
            psi.factor = 1 if normalize else psi.factor * ntmp
            if ntmp:
                tmp = (tmp / ntmp)
            psi[n] = tmp.transpose(axes=(1, 0, 2))

    return psi, discarded2_total ** 0.5
