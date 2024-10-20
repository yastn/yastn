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
""" Methods to generate (Hamiltonian) Mpo. """
from __future__ import annotations
from operator import itemgetter
import numpy as np
import numbers
from typing import NamedTuple
from ... import zeros, ncon, Leg, YastnError, Tensor, block, svd_with_truncation, allclose
from ._mps_obc import Mpo
from ...operators import sign_canonical_order
from itertools import groupby


class Hterm(NamedTuple):
    r"""
    Defines a product operator :math:`O = amplitude{\times}\bigotimes_i o_i` of local operators :math:`o_i`.
    Local operators that are not explicitly specified are assumed to be identity operators.

    If operators are fermionic, execution of :meth:`swap gates<yastn.swap_gate>` enforces fermionic order,
    with the last operator in `operators` acting first.

    Parameters
    ----------
    amplitude: number
        numerical multiplier in front of the operator product.
    positions: Sequence[int]
        positions of the local operators :math:`o_i` in the product different than identity.
    operators: Sequence[yastn.Tensor]
        local operators in the product that are different than the identity.
        *i*-th operator is acting at ``positions[i]``.
        Each operator should have ``ndim=2`` and signature ``s=(+1, -1)``.
    """
    amplitude: float = 1.0
    positions: tuple = ()
    operators: tuple = ()


def ind_list_tensors(op: Tensor, unique: list) -> int:
    """ Return index of op in unique. If not present, append op to unique. """
    ind = next((ind for ind, v in enumerate(unique) if v is op or allclose(v, op)), None)
    if ind is None:
        ind = len(unique)
        unique.append(op)
    return ind


def ind_list(el, unique):
    """ Return index of hashable el in unique. If not present, append el to unique. """
    try:
        ind = unique.index(el)
    except ValueError:
        ind = len(unique)
        unique.append(el)
    return ind


def generate_mpo(I, terms=None, opts_svd=None) -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Generate MPO provided a list of :class:`Hterm`\-s and identity MPO ``I``.

    It employs :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`
    and :meth:`mps.generate_mpo_fast<yastn.tn.mps.generate_mpo_fast>`,
    but without bookkeeping the template obtained in the preprocessing step.
    The latter would speed up the generation of MPOs for different amplitudes
    in front of the same set of product operators.

    Parameters
    ----------
    terms: Sequence[yastn.tn.mps.Hterm]
        Product operators making up MPO.
    I: yastn.tn.mps.MpsMpoOBC
        Identity MPO.
    opts: dict
        Options passed to :meth:`yastn.linalg.svd_with_truncation`.
        The function employs SVD while compressing the MPO bond dimensions.
        Default ``None`` sets truncation ``tol`` close to the numerical precision,
        which typically results in lossless compression.
    """
    if not terms:
        return I.copy()

    try:
        if any(len(term.positions) != len(term.operators) for term in terms):
            raise YastnError("Hterm: numbers of positions and operators do not match. ")
    except TypeError:
        raise YastnError("Hterm: positions and operators should be provided as lists or tuples.")

    unique_ops = []
    Iind = [ind_list_tensors(I[n], unique_ops) for n in I.sweep()]
    unique_ops = [op.remove_leg(axis=0).remove_leg(axis=1) for op in unique_ops]
    Iop = [unique_ops[k] for k in Iind]

    M, N = len(terms), len(Iop)
    config = unique_ops[0].config
    sym = config.sym

    # generator will assume operators in each term to be in fermionic order,
    # i.e., operators at later sites in the chain are applied first
    # sign to permute to canonical order is calculated in signs
    f_ordered = lambda s0, s1: s0 <= s1
    signs, sitess, opss, op_patterns = [], [], [], []
    for term in terms:
        if any(site < 0 or site > N or not isinstance(site, numbers.Integral) for site in term.positions):
            raise YastnError("position in Hterm should be in 0, 1, ..., N-1")
        if any(op.s != Iop[site].s for op, site in zip(term.operators, term.positions)):
            raise YastnError("operator in Hterm should be a matrix with signature matching I at given site")

        signs.append(sign_canonical_order(*term.operators, sites=term.positions, f_ordered=f_ordered))
        sites_ops = sorted(zip(term.positions, term.operators), key=itemgetter(0))
        sites, ops = [], []
        for site, group in groupby(sites_ops, key=itemgetter(0)):
            sites.append(site)
            op = next(group)[1]
            for el in group:
                op = op @ el[1]
            ops.append(ind_list_tensors(op, unique_ops))
        sites.append(N)
        sitess.append(sites)
        opss.append(ind_list(ops, op_patterns))

    n_patterns = [[unique_ops[ind].n for ind in ops] for ops in op_patterns]
    n_patterns = [[sym.add_charges(*ns[n:]) for n in range(len(ns) + 1)] for ns in n_patterns]

    # encoding local operators for each term and each site in range(N)
    # include information about charges connecing local operators
    mapH = np.zeros((M, N), dtype=np.int64)
    charges_ops = []
    for mH, sites, ind in zip(mapH, sitess, opss):
        ops, n_pattern = op_patterns[ind], n_patterns[ind]
        ii = 0
        site, tl = sites[ii], n_pattern[ii]
        for n in range(N):
            if n == site:
                op, tr = ops[ii], n_pattern[ii + 1]
                mH[n] = ind_list((tl, op, tr), charges_ops)
                ii += 1
                site, tl = sites[ii], n_pattern[ii]
            else:
                mH[n] = ind_list((tl, Iind[n], tl), charges_ops)

    # turn 2-leg local operators into 4-legs local operators that will build MPO
    dressed_ops = []
    for tl, ind, tr in charges_ops:
        op = unique_ops[ind]
        op = op.swap_gate(axes=1, charge=tr)
        op = op.add_leg(axis=1, s=1, t=tr)
        op = op.add_leg(axis=0, s=-1, t=tl)
        dressed_ops.append(op)
        assert op.n == sym.zero(), "Should not have happen if charges are correctly added."

    # is a product state
    if M == 1:
        mH = mapH[0]
        O = Mpo(N)
        for n in O.sweep():
            O[n] = dressed_ops[mH[n]]
        O[0] = O[0] * (terms[0].amplitude * signs[0])
        return O

    basis, t1bs, t2bs, tfbs, ifbs = [], [], [], [], []
    for n in range(N):
        un = sorted(np.unique(mapH[:, n]).tolist())
        t1bs.append({k: charges_ops[k][0] for k in un})
        t2bs.append({k: charges_ops[k][2] for k in un})
        tfbs.append({k: unique_ops[charges_ops[k][1]].n for k in un})
        tmp, ifb = {}, {}
        for k, t in tfbs[-1].items():
            ifb[k] = tmp.get(t, 0)
            tmp[t] = ifb[k] + 1
        ifbs.append(ifb)

        base = {k: dressed_ops[k].fuse_legs(axes=((0, 2), 1, 3)).drop_leg_history() for k in un}
        basis.append(block(base, common_legs=(1, 2)).drop_leg_history())

    tleft = set(t1bs[0].values())
    if len(tleft) != 1:
        raise YastnError("generate_mpo: provided terms do not all have the same total charge.")
    tleft = tleft.pop()

    trans = []
    for n in I.sweep():
        mapH0 = mapH[:, 0].tolist()
        mapH, rind, iind = np.unique(mapH[:, 1:], axis=0, return_index=True, return_inverse=True)
        iind = iind.ravel().tolist()
        rind = rind.ravel().tolist()

        i2bs = {t: {} for t in t2bs[n].values()}
        for ii, rr in enumerate(rind):
            i2bs[t2bs[n][mapH0[rr]]][ii] = len(i2bs[t2bs[n][mapH0[rr]]])

        i1bs = {t: 0 for t in t1bs[n].values()}
        for ii, rr in enumerate(mapH0):
            i1bs[t1bs[n][rr]] += 1

        leg1 = Leg(config, s=-1, t=list(i1bs.keys()), D=list(i1bs.values()))
        leg2 = basis[n].get_legs(axes=0).conj()
        leg3 = Leg(config, s=1, t=list(i2bs.keys()), D=[len(x) for x in i2bs.values()])
        tran = zeros(config=config, legs=[leg1, leg2, leg3])

        li = {x: -1 for x in t1bs[n].values()}
        for bl, br in zip(mapH0, iind):
            lt = t1bs[n][bl]
            li[lt] += 1
            ft = tfbs[n][bl]
            fi = ifbs[n][bl]
            rt = t2bs[n][bl]
            ri = i2bs[rt][br]
            tran[lt + ft + rt][li[lt], fi, ri] += 1
        trans.append(tran)

    amplitudes = [term.amplitude * sign for term, sign in zip(terms, signs)]
    dtype = 'complex128' if any(isinstance(a, complex) for a in amplitudes) else config.default_dtype
    J = Tensor(config=config, s=(-1, 1), dtype=dtype)
    J.set_block(ts=(tleft, tleft), Ds=(1, M), val=amplitudes)

    if opts_svd is None:
        opts_svd = {'tol': 1e-13}

    O = Mpo(N)
    for n in O.sweep():
        nJ = J @ trans[n]
        nJ = ncon([nJ, basis[n]], [[0, 1, -3], [1, -1, -2]])
        if n < O.last:
            nJ, S, V = svd_with_truncation(nJ, axes=((0, 1, 2), 3), sU=1, **opts_svd)
            nS = S.norm()
            nJ = nS * nJ
            J = (S / nS) @ V
        O[n] = nJ.transpose(axes=(0, 1, 3, 2))
    return O


# def generate_product_mpo_from_Hterm(I, term, amplitude=True) -> yastn.tn.mps.MpsMpoOBC:
#     r"""
#     Apply local operators specified by term in :class:`Hterm` to
#     the list of identities `I`. Generate product MPO as a result.
#     As an input, insted of MPO, we take the list of identities for simpler manipulations.

#     MPO `I` is presumed to be an identity.
#     Apply swap_gates to introduce fermionic degrees of freedom
#     (fermionic order is the same as the order of sites in Mps).
#     With this respect, local operators specified in term.operators are applied starting with the last element,
#     i.e., from right to left.

#     Parameters
#     ----------
#     term: yastn.tn.mps.Hterm

#     amplitude: bool
#         if True, includes term.amplitude in MPO.
#     """
#     try:
#         if len(term.positions) != len(term.operators):
#             raise YastnError("Hterm: numbers of positions and operators do not match. ")
#     except TypeError:
#         raise YastnError("Hterm: positions and operators should be provided as lists or tuples.")

#     tmp = I.copy()
#     for site, op in zip(term.positions[::-1], term.operators[::-1]):
#         if site < 0 or site > len(I) or not isinstance(site, numbers.Integral):
#             raise YastnError("position in Hterm should be in 0, 1, ..., N-1 ")
#         if op.s != I[site].s:
#             raise YastnError("operator in Hterm should be a matrix with signature matching I")

#         tmp[site] = op @ tmp[site]
#         charge = op.n
#         for n in range(site):
#             tmp[n] = tmp[n].swap_gate(axes=0, charge=charge)

#     psi = Mpo(len(I))
#     rt = op.config.sym.zero()
#     for n, vec in zip(psi.sweep(to='first'), tmp[::-1]):
#         vec = vec.add_leg(axis=1, s=1, t=rt)
#         rt = vec.n
#         psi[n] = vec.add_leg(axis=0, s=-1)

#     if amplitude: psi[0] = term.amplitude * psi[0]

#     return psi


# class GenerateMpoTemplate(NamedTuple):
#     config: NamedTuple = None
#     basis: list = None
#     trans: list = None
#     tleft: list = None


# def generate_mpo_preprocessing(I, terms=None) -> GenerateMpoTemplate | tuple[GenerateMpoTemplate, list[float]]:
#     r"""
#     Precompute an amplitude-independent template that is used
#     to generate MPO with :meth:`mps.generate_mpo_fast<yastn.tn.mps.generate_mpo_fast>`

#     Parameters
#     ----------
#     I: yastn.tn.mps.MpsMpoOBC
#         identity MPO.

#     terms: list of :class:`Hterm`
#         product operators making up the MPO.
#     """
#     if terms is None or len(terms) == 0:
#         return GenerateMpoTemplate(trans=I)
#     I2 = [I[n].remove_leg(axis=0).remove_leg(axis=1) for n in I.sweep(to='last')]
#     H1s = [generate_product_mpo_from_Hterm(I2, term, amplitude=False) for term in terms]
#     cfg = H1s[0][0].config
#     mapH = np.zeros((len(H1s), I.N), dtype=np.int64)

#     basis, t1bs, t2bs, tfbs, ifbs = [], [], [], [], []
#     for n in I.sweep():
#         base = []
#         for m, H1 in enumerate(H1s):
#             # site-tensors for which yastn.allclose is True are assumed idential.
#             ind = next((ind for ind, v in enumerate(base) if allclose(v, H1[n])), None)
#             if ind is None:
#                 ind = len(base)
#                 base.append(H1[n])
#             mapH[m, n] = ind

#         t1bs.append([ten.get_legs(axes=0).t[0] for ten in base])
#         t2bs.append([ten.get_legs(axes=2).t[0] for ten in base])
#         base = [ten.fuse_legs(axes=((0, 2), 1, 3)).drop_leg_history() for ten in base]
#         tfb = [ten.get_legs(axes=0).t[0] for ten in base]
#         tfbs.append(tfb)
#         ifbs.append([sum(x == y for x in tfb[:i]) for i, y in enumerate(tfb)])
#         base = block(dict(enumerate(base)), common_legs=(1, 2)).drop_leg_history()
#         basis.append(base)

#     tleft = [t1bs[0][i] for i in mapH[:, 0].tolist()]

#     trans = []
#     for n in I.sweep():
#         mapH0 = mapH[:, 0].tolist()
#         mapH, rind, iind = np.unique(mapH[:, 1:], axis=0, return_index=True, return_inverse=True)
#         iind = iind.ravel().tolist()
#         rind = rind.ravel().tolist()

#         i2bs = {t: {} for t in t2bs[n]}
#         for ii, rr in enumerate(rind):
#             i2bs[t2bs[n][mapH0[rr]]][ii] = len(i2bs[t2bs[n][mapH0[rr]]])

#         i1bs = {t: 0 for t in t1bs[n]}
#         for ii, rr in enumerate(mapH0):
#             i1bs[t1bs[n][rr]] += 1

#         leg1 = Leg(cfg, s=-1, t=list(i1bs.keys()), D=list(i1bs.values()))
#         leg2 = basis[n].get_legs(axes=0).conj()
#         leg3 = Leg(cfg, s=1, t=list(i2bs.keys()), D=[len(x) for x in i2bs.values()])
#         tran = zeros(config=cfg, legs=[leg1, leg2, leg3])

#         li = {x: -1 for x in t1bs[n]}
#         for bl, br in zip(mapH0, iind):
#             lt = t1bs[n][bl]
#             li[lt] += 1
#             ft = tfbs[n][bl]
#             fi = ifbs[n][bl]
#             rt = t2bs[n][bl]
#             ri = i2bs[rt][br]
#             tran[lt + ft + rt][li[lt], fi, ri] += 1
#         trans.append(tran)

#     return GenerateMpoTemplate(config=cfg, basis=basis, trans=trans, tleft=tleft)


# def generate_mpo_fast(template, amplitudes, opts_svd=None) -> yastn.tn.mps.MpsMpoOBC:
#     r"""
#     Fast generation of MPOs representing `Sequence[Hterm]` that differ only in amplitudes.

#     Preprocessing in :meth:`yastn.tn.mps.generate_mpo` might be slow.
#     When only amplitudes in Hterms are changing, e.g., for time-dependent Hamiltonian,
#     MPO generation can be significantly speeded up by precalculating and reusing amplitude-independent ``template``.
#     The latter is done with :meth:`yastn.tn.mps.generate_mpo_preprocessing`.

#     Parameters
#     ----------
#     template: NamedTuple
#         Calculated with :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`.
#     amplidutes: Sequence[numbers]
#         List of amplitudes that would appear in :class:`Hterm`.
#         The order of the list should match the order of `Sequence[Hterm]`
#         supplemented to :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`.
#     opts: dict
#         Options passed to :meth:`yastn.linalg.svd_with_truncation`.
#         The function employs SVD while compressing the MPO bond dimensions.
#         Default ``None`` sets truncation ``tol`` close to the numerical precision,
#         which typically results in lossless compression.
#     """
#     if len(amplitudes) == 0:
#         return template.trans.copy()

#     if opts_svd is None:
#         opts_svd = {'tol': 1e-13}

#     Js = {}
#     for a, t in zip(amplitudes, template.tleft):
#         if t in Js:
#             Js[t].append(a)
#         else:
#             Js[t] = [a]

#     dtype = 'complex128' if any(isinstance(a, complex) for a in amplitudes) else template.config.default_dtype
#     J = Tensor(config=template.config, s=(-1, 1), dtype=dtype)
#     for t, val in Js.items():
#         J.set_block(ts=(t, t), Ds=(1, len(val)), val=val)

#     M = Mpo(len(template.basis))
#     for n in M.sweep():
#         nJ = J @ template.trans[n]
#         nJ = ncon([nJ, template.basis[n]], [[0, 1, -3], [1, -1, -2]])
#         if n < M.last:
#             nJ, S, V = svd_with_truncation(nJ, axes=((0, 1, 2), 3), sU=1, **opts_svd)
#             nS = S.norm()
#             nJ = nS * nJ
#             J = (S / nS) @ V
#         M[n] = nJ.transpose(axes=(0, 1, 3, 2))
#     return M


# def generate_mpo(I, terms=None, opts_svd=None) -> yastn.tn.mps.MpsMpoOBC:
#     r"""
#     Generate MPO provided a list of :class:`Hterm`\-s and identity MPO ``I``.

#     It employs :meth:`mps.generate_mpo_preprocessing<yastn.tn.mps.generate_mpo_preprocessing>`
#     and :meth:`mps.generate_mpo_fast<yastn.tn.mps.generate_mpo_fast>`,
#     but without bookkeeping the template obtained in the preprocessing step.
#     The latter would speed up the generation of MPOs for different amplitudes
#     in front of the same set of product operators.

#     Parameters
#     ----------
#     terms: Sequence[yastn.tn.mps.Hterm]
#         Product operators making up MPO.
#     I: yastn.tn.mps.MpsMpoOBC
#         Identity MPO.
#     opts: dict
#         Options passed to :meth:`yastn.linalg.svd_with_truncation`.
#         The function employs SVD while compressing the MPO bond dimensions.
#         Default ``None`` sets truncation ``tol`` close to the numerical precision,
#         which typically results in lossless compression.
#     """
#     template = generate_mpo_preprocessing(I, terms)
#     amplitudes = [term.amplitude for term in terms]
#     return generate_mpo_fast(template, amplitudes, opts_svd=opts_svd)
