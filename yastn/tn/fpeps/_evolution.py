# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
""" Routines for time evolution with nn gates on a 2D lattice. """
from itertools import pairwise
from typing import NamedTuple

import numpy as np

from ._gates_auxiliary import Gate, gate_from_mpo
from ._peps import Peps2Layers
from ..mps import MpsMpoOBC
from ...tensor import tensordot, vdot, svd_with_truncation, YastnError, Tensor



class BondMetric(NamedTuple):
    g: Tensor = None


class BipartiteBondMetric(NamedTuple):
    gL: Tensor = None
    gR: Tensor = None


class Evolution_out(NamedTuple):
    """ All errors and eigenvalues are relative. """
    bond: tuple = None
    truncation_error: float = 0
    best_method: str = ''
    nonhermitian_part: float = 0
    min_eigenvalue: float = None
    wrong_eigenvalues: float = None
    eat_metric_error: float = None
    truncation_errors: dict[str, float] = ()
    iterations: dict[str, int] = ()
    pinv_cutoffs: dict[str, float] = ()
    loopiness: float = 0
    truncated_sectors:dict[str, tuple] = ()


def evolution_step_(env, gates, opts_svd, method='mpo', fix_metric=0,
                    pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                    max_iter=100, tol_iter=1e-13, initialization="EAT_SVD"):
    r"""
    Perform a single step of PEPS evolution by applying a list of gates.
    Truncate bond dimension after each application of a two-site gate.

    Parameters
    ----------
    env: EnvNTU | EnvCTM | EnvApproximate
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    gates: yastn.tn.fpeps.Gates
        The gates to be applied to PEPS.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    method: str
        For ``'NN'``, split multi-site gates into a series of nearest-neighbor gates.
        Otherwise, apply mpo-gate first, and then sequentially truncate enlarged bonds (the default).
    fix_metric: int | None
        Error measure of the metric tensor is a sum of: the norm of its non-hermitian part
        and absolute value of the most negative eigenvalue (if present).
        If ``fix_metric`` is a number, replace eigenvalues smaller than the error_measure with fix_metric * error_measure.
        Sensible values of ``fix_metric`` are :math:`0` and :math:`1`. The default is :math:`0`.
        If ``None``, do not perform eigh to test for negative eigenvalues and do not fix metric.
    pinv_cutoffs: Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimizations and EAT initialization.
    max_iter: int
        The maximal number of iterative steps for each truncation optimization.
    tol_iter: int
        Tolerance of truncation_error to stop iterative optimization.
    initialization: str
        Tested initializations of iterative optimization. The one resulting in the smallest error is selected.
        Possible options are 'SVD' (svd initialization only), 'EAT' (EAT optimization only), 'SVD_EAT' (tries both).

    Returns
    -------
    list[Evolution_out(NamedTuple)]
        List of Namedtuple containing fields:
            * ``bond`` bond where the gate is applied.
            * ``truncation_error`` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env.
            * ``best_method`` initialization/optimization method giving the best truncation_error. Possible values are 'eat', 'eat_opt', 'svd', 'svd_opt'.
            * ``nonhermitian_part`` norm of the non-hermitian part of the bond metric, normalized by the bond metric norm. Estimator of metric error.
            * ``min_eigenvalue`` the smallest bond metric eigenvalue, normalized by the bond metric norm. Can be negative which gives an estimate of metric error.
            * ``wrong_eigenvalues`` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to ``fix_metric`` argument.
            * ``eat_metric_error`` error of approximating metric tensor by a product via SVD-1 in EAT initialization.
            * ``truncation_errors`` dict with truncation_error-s for all tested initializations/optimizations.
            * ``iterations`` dict with number of iterations to converge iterative optimization.
            * ``pinv_cutoffs`` dict with optimal pinv_cutoffs for methods where pinv appears.
    """
    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    infos = []

    if 'nn' in method.lower():
        gates = [Gate(gate_from_mpo(gate.G), gate.sites) if isinstance(gate.G, MpsMpoOBC) else gate  for gate in gates]
        gates = [ng for og in gates for ng in split_gate_2site(og)]

    for gate in gates:
        psi.apply_gate_(gate)

        env.pre_truncation_(gate.sites)
        for s0, s1 in pairwise(gate.sites):
            info = truncate_(env, opts_svd, (s0, s1), fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization)
            infos.append(info)

    return infos


def split_gate_2site(gate):
    """
    Split gate-mpo into a series of 2-site gates using SVD.
    """
    if len(gate.G) < 3:
        return [gate]
    new_gates = []
    g0, s0 = gate.G[0], gate.sites[0]
    for g1, s1 in zip(gate.G[1:-1], gate.sites[1:-1]):
        U, S, V = g1.svd_with_truncation(axes=((1, 2), (0, 3)), Uaxis=0, Vaxis=1, tol=1e-14)
        S = S.sqrt()
        U, V = S.broadcast(U, V, axes=(0, 1))
        new_gates.append(Gate(G=(g0, U), sites=(s0, s1)))
        g0, s0 = V, s1
    new_gates.append(Gate(G=(g0, gate.G[-1]), sites=(s0, gate.sites[-1])))
    return new_gates


def truncate_(env, opts_svd, bond=None,
              fix_metric=0,
              pinv_cutoffs=(1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
              max_iter=100, tol_iter=1e-13, initialization="EAT_SVD"):
    r"""
    Truncate virtual bond dimensions of PEPS.

    Parameters
    ----------
    env: EnvNTU | EnvCTM | EnvApproximate
        Environment class containing PEPS state (updated in place),
        and a method to calculate bond metric tensors employed during truncation.
    opts_svd: dict | Sequence[dict]
        Options passed to :meth:`yastn.linalg.svd_with_truncation` which are used
        to initially truncate the bond dimension before it is further optimized iteratively.
        In particular, it fixes bond dimensions (potentially, sectorial).
        It is possible to provide a list of dicts (with decreasing bond dimensions),
        in which case the truncation is done gradually in a few steps.
    bond: tuple[Site, Site] | None
        Specify single bond to be truncated. If None, truncate all bonds. The default is None.
    fix_metric: int | None
        Error measure of the metric tensor is a sum of: the norm of its non-hermitian part
        and absolute value of the most negative eigenvalue (if present).
        If ``fix_metric`` is a number, replace eigenvalues smaller than the error_measure with fix_metric * error_measure.
        Sensible values of ``fix_metric`` are :math:`0` and :math:`1`. The default is :math:`0`.
        If ``None``, do not perform eigh to test for negative eigenvalues and do not fix metric.
    pinv_cutoffs: Sequence[float] | float
        List of pseudo-inverse cutoffs.
        The one that gives the smallest truncation error is used during iterative optimizations and EAT initialization.
    max_iter: int
        The maximal number of iterative steps for each truncation optimization.
    tol_iter: int
        Tolerance of truncation_error to stop iterative optimization.
    initialization: str
        Tested initializations of iterative optimization. The one resulting in the smallest error is selected.
        Possible options are 'SVD' (svd initialization only), 'EAT' (EAT optimization only), 'SVD_EAT' (tries both).

    Returns
    -------
    Evolution_out(NamedTuple)
        Namedtuple containing fields:
            * ``bond`` bond where the gate is applied.
            * ``truncation_error`` relative norm of the difference between untruncated and truncated bonds, calculated in metric specified by env.
            * ``best_method`` initialization/optimization method giving the best truncation_error. Possible values are 'eat', 'eat_opt', 'svd', 'svd_opt'.
            * ``nonhermitian_part`` norm of the non-hermitian part of the bond metric, normalized by the bond metric norm. Estimator of metric error.
            * ``min_eigenvalue`` the smallest bond metric eigenvalue, normalized by the bond metric norm. Can be negative which gives an estimate of metric error.
            * ``wrong_eigenvalues`` a fraction of bond metrics eigenvalues that were below the error threshold; and were modified according to ``fix_metric`` argument.
            * ``eat_metric_error`` error of approximating metric tensor by a product via SVD-1 in EAT initialization.
            * ``truncation_errors`` dict with truncation_error-s for all tested initializations/optimizations.
            * ``iterations`` dict with number of iterations to converge iterative optimization.
            * ``pinv_cutoffs`` dict with optimal pinv_cutoffs for methods where pinv appears.
    """
    psi = env.psi
    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    if bond is None:
        bonds = psi.bonds()
    else:
        bonds = [bond]

    infos = []
    for bond in bonds:
        dirn = psi.nn_bond_dirn(*bond)
        if dirn in ('rl', 'bt'):
            bond, dirn = bond[::-1], dirn[::-1]  # 'lr' or 'tb'

        s0, s1 = bond
        info = {'bond': bond}

        if dirn == 'lr':  # Horizontal gate, "lr" ordered
            Q0, R0 = psi[s0].qr(axes=((0, 1, 2, 4), 3), sQ=-1, Qaxis=3)  # t l b rr sa @ rr r
            Q1, R1 = psi[s1].qr(axes=((0, 2, 3, 4), 1), sQ=1, Qaxis=1, Raxis=-1)  # t ll b r sa @ l ll
        else:  # dirn == 'tb':  Vertical gate, "tb" ordered
            Q0, R0 = psi[s0].qr(axes=((0, 1, 3, 4), 2), sQ=1, Qaxis=2)  # t l bb r sa @ bb b
            Q1, R1 = psi[s1].qr(axes=((1, 2, 3, 4), 0), sQ=-1, Qaxis=0, Raxis=-1)  # tt l b r sa @ t tt

        fgf = env.bond_metric(Q0, Q1, s0, s1, dirn)

        if isinstance(fgf, BipartiteBondMetric):  # bipartite bond metric
            M0, M1, info = truncate_bipartite_(fgf, R0, R1, opts_svd, pinv_cutoffs, info)
        elif isinstance(fgf, BondMetric):  # bipartite bond metric
            M0, M1, info = truncate_optimize_(fgf, R0, R1, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization, info)

        if dirn == 'lr':
            psi[s0] = tensordot(Q0, M0, axes=(3, 0)).transpose(axes=(0, 1, 2, 4, 3))  # t l b r sa
            psi[s1] = tensordot(M1, Q1, axes=(1, 1)).transpose(axes=(1, 0, 2, 3, 4))  # t l b r sa
        else:  # dirn == 'tb':
            psi[s0] = tensordot(Q0, M0, axes=(2, 0)).transpose(axes=(0, 1, 4, 2, 3))  # t l b r sa
            psi[s1] = tensordot(M1, Q1, axes=(1, 0)) # t l b r sa

        env.post_truncation_(bond)
        infos.append(Evolution_out(**info))
    return infos[0] if len(bonds) == 1 else infos

def build_g_rj(r_slices: dict, G0: Tensor):

    g = []
    for key1 in r_slices.keys():
        for ii in range(len(r_slices[key1])):
            g.append([])
            for key2 in r_slices.keys():
                for jj in range(len(r_slices[key2])):
                    temp = tensordot(r_slices[key1][ii], G0, axes=((0, 1), (2, 3)))
                    temp = tensordot(r_slices[key2][jj], temp, axes=((0, 1), (0, 1)), conj=(1, 0))._data[0]
                    g[len(g) - 1].append(temp)
    g = np.array(g)
    g = (g + g.T.conjugate()) / 2
    return g

def initial_truncation_ZMT1(R0, R1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial=None):

    if opts_svd.get("preD") is None:
        preD = 32767
    else:
        preD = opts_svd["preD"]

    if pre_initial == "EAT":
        (R0, R1), _, _, _= initial_truncation_EAT(R0, R1, fgf, fRR, RRgRR, {"D_total": preD, "tol":-1}, pinv_cutoffs)
    elif pre_initial == "SVD":
        R0, S, R1 = svd_with_truncation(R0 @ R1, sU=R0.s[1], D_total=preD)
        S = S.sqrt()
        R0, R1 = S.broadcast(R0, R1, axes=(1, 0))

    G0 = fgf.unfuse_legs(axes=(0, 1))

    G = fgf.unfuse_legs(axes=1)
    G = tensordot(G, R0, axes=(1, 0))
    G = tensordot(G, R1, axes=(1, 1))
    G = G.fuse_legs(axes=((0, (1, 2))))
    G = G.unfuse_legs(axes=0)
    G = tensordot(R1.conj(), G, axes=(1, 1))
    G = tensordot(R0.conj(), G, axes=(0, 1))
    Gremove = G.unfuse_legs(axes=2)
    Gremove.remove_zero_blocks()
    Gremove = Gremove.fuse_legs(axes=((0, 2), (1, 3)))
    _, S, _ = svd_with_truncation(Gremove, axes=(0, 1), policy='lowrank', D_block=2, D_total=2)
    S = np.diag(S.to_numpy())
    loopiness = np.min(S) / np.max(S)

    # slice RA to column vectors
    data_r0 = R0.T.compress_to_1d()
    accumulated = 0
    r0_slices = {}
    D_total = 0
    len_t = len(data_r0[1]['struct'].t[0]) // 2
    for ii in range(len(data_r0[1]['struct'].D)):
        r0_slices[data_r0[1]['struct'].t[ii]] = []
        Ds = data_r0[1]['struct'].D[ii]
        D_total = D_total + Ds[0]
        for _ in range(Ds[0]):
            data = data_r0[0][accumulated:(accumulated + Ds[1])]
            tensor = Tensor(config=R0.config, s=R0.T.get_signature(), dtype="complex128")
            tensor.set_block(ts=(data_r0[1]['struct'].t[ii][0:len_t], data_r0[1]['struct'].t[ii][len_t:]), val=data, Ds=[1, Ds[1]])
            r0_slices[data_r0[1]['struct'].t[ii]].append(tensor.T)
            accumulated = accumulated + Ds[1]

    # slice RB to row vectors
    data_r1 = R1.compress_to_1d()
    accumulated = 0
    r1_slices = {}
    for ii in range(len(data_r1[1]['struct'].D)):
        r1_slices[data_r1[1]['struct'].t[ii]] = []
        Ds = data_r1[1]['struct'].D[ii]
        for _ in range(Ds[0]):
            data = data_r1[0][accumulated:(accumulated + Ds[1])]
            tensor = Tensor(config=R0.config, s=R1.get_signature(), dtype="complex128")
            tensor.set_block(ts=(data_r1[1]['struct'].t[ii][0:len_t], data_r1[1]['struct'].t[ii][:len_t]), val=data, Ds=[1, Ds[1]])
            r1_slices[data_r1[1]['struct'].t[ii]].append(tensor)
            accumulated = accumulated + Ds[1]
    # build Rj=RAj * RBj
    r_slices = {}
    for ii in range(len(data_r0[1]['struct'].D)):
        r0s = r0_slices[data_r0[1]['struct'].t[ii]]
        r1s = r1_slices.get(data_r0[1]['struct'].t[ii])
        if r1s is not None:
            r_slices[data_r0[1]['struct'].t[ii]] = []
            for kk in range(len(r0s)):
                r0 = r0s[kk]
                r1 = r1s[kk]
                r_slices[data_r0[1]['struct'].t[ii]].append(r0 @ r1)

    weight = {}
    for ii in range(len(data_r0[1]['struct'].D)):
        Ds = data_r0[1]['struct'].D[ii]
        weight[data_r0[1]['struct'].t[ii]] = [1.0 + 0.0j for _ in range (Ds[0])]

    removed = 0
    while ((D_total - removed) > opts_svd['D_total']):
        g = build_g_rj(r_slices, G0)
        S, W = np.linalg.eigh(g)

        to_be_removed_w = 0
        to_be_eliminated = np.argmax(np.abs(W[:, to_be_removed_w].T))
        coef = 1

        # update weight & kick dropped index
        s = 0
        sum_weight2 = 0
        for key in r_slices.keys():
            eliminate_ii = None
            for ii in range(len(r_slices[key])):
                if s == to_be_eliminated:
                    eliminate_ii = ii
                else:
                    weight[key][ii] = 1 - (W[s,to_be_removed_w] / W[to_be_eliminated,to_be_removed_w]) * coef
                    sum_weight2 = sum_weight2 + np.abs(weight[key][ii]) ** 2
                s = s + 1

            if eliminate_ii is not None:
                weight[key].pop(eliminate_ii)
                r_slices[key].pop(eliminate_ii)
                r0_slices[key].pop(eliminate_ii)
                r1_slices[key].pop(eliminate_ii)

        for key in r_slices.keys():
            for ii in range(len(r_slices[key])):
                r_slices[key][ii] = r_slices[key][ii] * weight[key][ii]
                r0_slices[key][ii] = r0_slices[key][ii] * (weight[key][ii] ** 0.5)
                r1_slices[key][ii] = r1_slices[key][ii] * (weight[key][ii] ** 0.5)

        removed = removed + 1


    # Build MA and MB
    MA = Tensor(config=R0.config, s=R0.T.get_signature(), dtype="complex128")
    for key in r0_slices.keys():
        temp_block = []
        for ii in range(len(r0_slices[key])):
            temp_block.append(r0_slices[key][ii]._data)
        temp_block = np.array(temp_block)
        if len(temp_block) != 0:
            MA.set_block(ts=(r0_slices[key][ii].get_legs()[1].t[0], r0_slices[key][ii].get_legs()[0].t[0]),
                        Ds=temp_block.shape,
                        val=temp_block)
    MA = MA.T

    MB = Tensor(config=R1.config, s=R1.get_signature(), dtype="complex128")
    for key in r1_slices.keys():
        temp_block = []
        for ii in range(len(r1_slices[key])):
            temp_block.append(r1_slices[key][ii]._data)
        temp_block = np.array(temp_block)
        if len(temp_block) != 0:
            MB.set_block(ts=(r1_slices[key][ii].get_legs()[0].t[0], r1_slices[key][ii].get_legs()[1].t[0]),
                        Ds=temp_block.shape,
                        val=temp_block)

    MAMB = (MA @ MB)

    return (MA, MB), abs(calculate_truncation_error2(MAMB, fgf, fRR, RRgRR)), loopiness

def build_g_ijkl(fgf: Tensor, R0: Tensor, R1: Tensor):

    G = fgf.unfuse_legs(axes=1)
    G = tensordot(G, R0, axes=(1, 0))
    G = tensordot(G, R1, axes=(1, 1))
    G = G.fuse_legs(axes=((0, (1, 2))))
    G = G.unfuse_legs(axes=0)
    G = tensordot(R1.conj(), G, axes=(1, 1))
    G = tensordot(R0.conj(), G, axes=(0, 1))
    G = G.unfuse_legs(axes=2)
    # G = G.fuse_legs(axes=((0, 1), 2))

    gts = G.compress_to_1d()[1]['struct'].t
    slices = G.compress_to_1d()[1]['slices']
    gts_slices_dict = dict(zip(gts, slices))

    ts = G.get_legs()[0].t

    g = None
    for key1 in ts:
        temp = None
        for key2 in ts:
            target_key = (*key1, *key1, *key2, *key2)

            mat_begin = gts_slices_dict[target_key].slcs[0][0]
            mat_end = gts_slices_dict[target_key].slcs[0][1]
            mat_shape = [gts_slices_dict[target_key].D[0] * gts_slices_dict[target_key].D[1], gts_slices_dict[target_key].D[2] * gts_slices_dict[target_key].D[3]]
            mat = G._data[mat_begin:mat_end].reshape(mat_shape)
            if temp is None:
                temp = mat
            else:
                temp = np.hstack((temp, mat))
        if g is None:
            g = temp
        else:
            g = np.vstack((g, temp))

    return (g + g.T.conjugate()) / 2


def initial_truncation_ZMT3(R0, R1, fgf, opts_svd:dict, fRR, RRgRR, pinv_cutoffs, pre_initial=None):

    if opts_svd.get("preD") is None:
        preD = 32767
    else:
        preD = opts_svd["preD"]

    if pre_initial is None:
        MA = R0
        MB = R1
    elif pre_initial == "SVD":
        R0, S, R1 = svd_with_truncation(R0 @ R1, sU=R0.s[1], D_total=preD)
        S = S.sqrt()
        MA, MB = S.broadcast(R0, R1, axes=(1, 0))
    elif pre_initial == "EAT":
        (MA, MB), error2, _, _ = initial_truncation_EAT(R0, R1, fgf, fRR, RRgRR, {"D_total":preD, "tol":-1}, pinv_cutoffs)
    elif pre_initial[:4] == "ZMT1":
        opts_svd_pre = {'D_total': preD, 'preD':None, 'tol':-1}
        if pre_initial == "ZMT10":
            pre_initial_ = None
        elif pre_initial == "ZMT1eat":
            pre_initial_ = "EAT"
        elif pre_initial == "ZMT1svd":
            pre_initial_ = "SVD"
        (MA, MB), error2, _ = initial_truncation_ZMT1(R0, R1, fgf, opts_svd_pre, fRR, RRgRR, pinv_cutoffs, pre_initial=pre_initial_)

    G = fgf.unfuse_legs(axes=1)
    G = tensordot(G, R0, axes=(1, 0))
    G = tensordot(G, R1, axes=(1, 1))
    G = G.fuse_legs(axes=((0, (1, 2))))
    G = G.unfuse_legs(axes=0)
    G = tensordot(R1.conj(), G, axes=(1, 1))
    G = tensordot(R0.conj(), G, axes=(0, 1))
    Gremove = G.unfuse_legs(axes=2)
    Gremove.remove_zero_blocks()
    Gremove = Gremove.fuse_legs(axes=((0, 2), (1, 3)))
    _, S, _ = svd_with_truncation(Gremove, axes=(0, 1), policy='lowrank', D_block=2, D_total=2)
    S = np.diag(S.to_numpy())
    loopiness = np.min(S) / np.max(S)


    D_total = 0
    for ii in range(len(MB.get_legs()[0].D)):
        D_total = D_total + MB.get_legs()[0].D[ii]

    while D_total > opts_svd['D_total']:

        g_ijkl = build_g_ijkl(fgf, MA, MB)
        _, W = np.linalg.eigh(g_ijkl)

        accumulated = 0
        normalization = 0
        for ii in range(len(MB.get_legs()[0].D)):
            Ds = MB.get_legs()[0].D[ii]
            mat = np.array(W[accumulated:(accumulated + Ds * Ds),0]).reshape(Ds, Ds)
            mat = (mat.T.conjugate()) @ mat

            normalization = normalization + mat.trace()
            accumulated = accumulated + Ds * Ds
        normalization = normalization ** 0.5

        accumulated = 0
        all_d = np.array([])
        for ii in range(len(MB.get_legs()[0].D)):
            Ds = MB.get_legs()[0].D[ii]
            mat = np.array(W[accumulated:(accumulated + Ds * Ds),0]).reshape(Ds, Ds) / normalization
            d = np.linalg.eigvals(mat)
            all_d = np.hstack([all_d, d])
            accumulated = accumulated + Ds * Ds

        largest_d = all_d[np.argmax(np.abs(all_d))]

        R = Tensor(config=R0.config, s=MB.get_signature(), dtype="complex128")

        accumulated = 0
        for ii in range(len(MB.get_legs()[0].D)):
            Ds = MB.get_legs()[0].D[ii]
            mat = -np.array(W[accumulated:(accumulated + Ds * Ds),0]).reshape(Ds, Ds) / largest_d + np.eye(Ds, Ds)
            R.set_block((MB.get_legs()[0].t[ii], MB.get_legs()[0].t[ii]), val=mat, Ds=[Ds, Ds])
            accumulated = accumulated + Ds * Ds

        D_total = D_total - 1

        R.remove_zero_blocks()
        U, S, Vh = svd_with_truncation(R, sU=R.s[1], D_total=D_total)
        S = S.sqrt()
        U, Vh = S.broadcast(U, Vh, axes=(1, 0))
        MA = MA @ U
        MB = Vh @ MB

        D_total = 0
        for new_d in MA.get_legs()[1].D:
            D_total = D_total + new_d

    error2 = calculate_truncation_error2(MA @ MB, fgf, fRR, RRgRR)
    return (MA, MB), error2, loopiness


def truncate_optimize_(g, R0, R1, opts_svd, fix_metric, pinv_cutoffs, max_iter, tol_iter, initialization, info):
    # enforce hermiticity
    fgf = g.g
    fgfH = fgf.H
    nonhermitian = (fgf - fgfH).norm() / 2
    fgf = (fgf + fgfH) / 2
    #
    fgf_norm = fgf.norm()
    info['nonhermitian_part'] = (nonhermitian / fgf_norm).item()
    #
    # check metric tensor eigenvalues
    if fix_metric is not None:
        S, U = fgf.eigh(axes=(0, 1))
        smin = min(S._data)
        info['min_eigenvalue'] = (smin / fgf_norm).item()
        g_error = max(-smin, 0) + nonhermitian
        info['wrong_eigenvalues'] = sum(S._data < g_error).item() / len(S._data)
        S._data[S._data < g_error] = g_error * fix_metric
        fgf = U @ S @ U.H
    #
    fRR = (R0 @ R1).fuse_legs(axes=[(0, 1)])
    fgRR = fgf @ fRR
    RRgRR = abs(vdot(fRR, fgRR).item())
    #
    pinv_cutoffs = sorted(pinv_cutoffs)
    M0, M1 = R0, R1
    loopiness = None
    for opts in [opts_svd] if isinstance(opts_svd, dict) else opts_svd:

        truncated_sectors, Ms, error2s, pinvs, iters = {}, {}, {}, {}, {}

        if 'EAT' in initialization:
            key = 'eat'
            Ms[key], error2s[key], pinvs[key], info["eat_metric_error"] = initial_truncation_EAT(M0, M1, fgf, fRR, RRgRR, opts, pinv_cutoffs)
            key = 'eat_opt'
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['eat'], error2s['eat'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'SVD' in initialization:
            key = 'svd'
            Ms[key] = symmetrized_svd(M0, M1, opts, normalize=False)
            error2s[key] = calculate_truncation_error2(Ms[key][0] @ Ms[key][1], fgf, fRR, RRgRR)
            key = 'svd_opt'
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['svd'], error2s['svd'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT10' in initialization:
            key = "ZMT1"
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT1(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs)
            key = "ZMT1_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT1'], error2s['ZMT1'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT1svd' in initialization:
            key = "ZMT1svd"
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT1(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="SVD")
            key = "ZMT1svd_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT1svd'], error2s['ZMT1svd'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT1eat' in initialization:
            key = "ZMT1eat"
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT1(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="EAT")
            key = "ZMT1eat_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT1eat'], error2s['ZMT1eat'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT30' in initialization:
            key = 'ZMT30'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial=None)
            key = "ZMT30_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT30'], error2s['ZMT30'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT3eat' in initialization:
            key = 'ZMT3eat'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="EAT")
            key = "ZMT3eat_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT3eat'], error2s['ZMT3eat'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT3svd' in initialization:
            key = 'ZMT3svd'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="SVD")
            key = "ZMT3svd_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT3svd'], error2s['ZMT3svd'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT3zmt10' in initialization:
            key = 'ZMT3zmt10'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="ZMT10")
            key = "ZMT3zmt10_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT3zmt10'], error2s['ZMT3zmt10'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT3zmt1eat' in initialization:
            key = 'ZMT3zmt1eat'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="ZMT1eat")
            key = "ZMT3zmt1eat_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT3zmt1eat'], error2s['ZMT3zmt1eat'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D
        if 'ZMT3zmt1svd' in initialization:
            key = 'ZMT3zmt1svd'
            Ms[key], error2s[key], loopiness = initial_truncation_ZMT3(M0, M1, fgf, opts_svd, fRR, RRgRR, pinv_cutoffs, pre_initial="ZMT1svd")
            key = "ZMT3zmt1svd_opt"
            Ms[key], error2s[key], pinvs[key], iters[key] = optimize_truncation(*Ms['ZMT3zmt1svd'], error2s['ZMT3zmt1svd'], fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter)
            truncated_sectors[key] = Ms[key][0].get_legs()[1].D

        if len(Ms) == 0:
            raise YastnError(f"{initialization=} not recognized. Should contain 'SVD' or 'EAT'.")

        error2s = {k: v ** 0.5 for k, v in error2s.items()}
        key = min(error2s, key=error2s.get)
        info['best_method'] = key
        info['truncation_errors'] = error2s
        info['truncation_error'] = error2s[key]
        info['pinv_cutoffs'] = pinvs
        info['iterations'] = iters
        info['loopiness'] = loopiness
        info['truncated_sectors'] = truncated_sectors
        M0, M1 = Ms[key]

    M0, M1 = symmetrized_svd(M0, M1, opts, normalize=True)
    return M0, M1, info


def truncate_bipartite_(g, R0, R1, opts_svd, pinv_cutoffs, info):
    #
    E0, E1 = g.gL, g.gR
    E0 = (E0 + E0.H) / 2
    E1 = (E1 + E1.H) / 2
    #
    RR = (R0 @ R1)
    gRR = E0 @ RR @ E1
    RRgRR = abs(vdot(RR, gRR).item())
    #
    F0 = R0.H @ E0 @ R0
    F1 = R1 @ E1 @ R1.H
    #
    S0, U0 = F0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    S1, U1 = F1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    #
    W0, W1 = symmetrized_svd(S0.sqrt() @ U0.H, U1 @ S1.sqrt(), opts_svd, normalize=False)
    p0, p1 = R0 @ U0, U1.H @ R1
    #
    error2, s0max, s1max = 100, max(S0._data), max(S1._data)
    vs0o, vs1o = len(S0._data) + 1, len(S1._data) + 1
    for c_off in sorted(pinv_cutoffs):
        vs0 = sum(S0._data > c_off * s0max).item()
        vs1 = sum(S1._data > c_off * s1max).item()
        if vs0 < vs0o or vs1 < vs1o:
            vs0o, vs1o = vs0, vs1
            M0_tmp = p0 @ S0.reciprocal(cutoff=c_off * s0max).sqrt() @ W0
            M1_tmp = W1 @ S1.reciprocal(cutoff=c_off * s1max).sqrt() @ p1
            delta = RR - M0_tmp @ M1_tmp
            error2_tmp = abs(vdot(delta, E0 @ delta @ E1).item()) / RRgRR
            if error2_tmp < error2:
                M0, M1, error2, pinv_c = M0_tmp, M1_tmp, error2_tmp, c_off

    info['best_method'] = 'bipartite'
    info['truncation_errors'] = {'bipartite': error2 ** 0.5}
    info['truncation_error'] = error2 ** 0.5
    info['pinv_cutoffs'] = pinv_c
    return M0, M1, info


def accumulated_truncation_error(infoss, statistics='mean'):
    r"""
    Return accumulated truncation error :math:`\Delta` calcuated from evolution output statistics.

    :math:`\Delta = \sum_{steps} statistics_{bond} [\sum_{gate \in bond} truncation\_error(gate, step)]`,
    where statistics is mean or max.

    Gives an estimate of errors accumulated during time evolution.

    Parameters
    ----------
    infoss: Sequence[Sequence[Evolution_out]]
        list of outputs of :meth:`evolution_step_`.
    statistics: str
        'max' or 'mean', whether to take the maximal value or a mean over the bonds in the lattice.

    Example
    -------

    ::

        infoss = []
        for step in range(number_of_steps):
            infos = fpeps.evolution_step_(env, gates, opt_svd)
            infoss.append(infos)

        # Accumulated runcation error
        Delta = fpeps.accumulated_truncation_error(infoss)
    """
    if statistics == 'max':
        stat = lambda x: max(x)
    elif statistics == 'mean':
        stat = lambda x: sum(x) / len(x)
    else:
        raise YastnError(f"{statistics=} in accumulated_truncation_error not recognized; Should be 'max' or 'mean'.")

    Delta = 0
    for infos in infoss:
        accum = {}
        for info in infos:
            accum[info.bond] = accum.get(info.bond, 0) + info.truncation_error
        Delta += stat(accum.values())
    return Delta


def symmetrized_svd(R0, R1, opts_svd, normalize=False):
    """ SVD truncation of central tensor; divide singular values symmetrically between tensors. """
    Q0, R0 = R0.qr(axes=(0, 1))
    Q1, R1 = R1.qr(axes=(1, 0), Qaxis=0, Raxis=1)
    U, S, V = svd_with_truncation(R0 @ R1, sU=R0.s[1], **opts_svd)
    if normalize:
        S = S / S.norm(p='inf')
    S = S.sqrt()
    M0, M1 = S.broadcast(U, V, axes=(1, 0))
    return Q0 @ M0, M1 @ Q1


def calculate_truncation_error2(fMM, fgf, fRR, RRgRR):
    """ Calculate squared truncation error. """
    if fMM.ndim == 2:  # if legs of MM not fused into vector
        fMM = fMM.fuse_legs(axes=[(0, 1)])
    delta = fRR - fMM
    return abs(vdot(delta, fgf @ delta).item()) / RRgRR


def initial_truncation_EAT(R0, R1, fgf, fRR, RRgRR, opts_svd, pinv_cutoffs):
    """
    Truncate R0 @ R1 to bond dimension specified in opts_svd
    including information from a product approximation of bond metric.
    """
    G = fgf.unfuse_legs(axes=(0, 1))
    #
    # rank-1 approximation
    Gremove = G.remove_zero_blocks()
    G0, S, G1 = svd_with_truncation(Gremove, axes=((0, 2), (3, 1)), policy='lowrank', D_block=1, D_total=1)
    fid = (S.norm() / G.norm()).item()
    eat_metric_error = (max(0., 1 - fid ** 2)) ** 0.5
    #
    G0 = G0.remove_leg(axis=2)
    G1 = G1.remove_leg(axis=0)
    #
    # make sure it is hermitian
    G0 = G0 / G0.trace().to_number()
    G1 = G1 / G1.trace().to_number()
    G0 = (G0 + G0.H) / 2
    G1 = (G1 + G1.H) / 2
    #
    F0 = R0.H @ G0 @ R0
    F1 = R1 @ G1 @ R1.H
    #
    S0, U0 = F0.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    S1, U1 = F1.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    #
    W0, W1 = symmetrized_svd(S0.sqrt() @ U0.H, U1 @ S1.sqrt(), opts_svd, normalize=False)
    p0, p1 = R0 @ U0, U1.H @ R1
    #
    error2, s0max, s1max = 100, max(S0._data), max(S1._data)
    vs0o, vs1o = len(S0._data) + 1, len(S1._data) + 1
    for c_off in pinv_cutoffs:
        vs0 = sum(S0._data > c_off * s0max).item()
        vs1 = sum(S1._data > c_off * s1max).item()
        if vs0 < vs0o or vs1 < vs1o:
            vs0o, vs1o = vs0, vs1
            M0_tmp = p0 @ S0.reciprocal(cutoff=c_off * s0max).sqrt() @ W0
            M1_tmp = W1 @ S1.reciprocal(cutoff=c_off * s1max).sqrt() @ p1
            error2_tmp = calculate_truncation_error2(M0_tmp @ M1_tmp, fgf, fRR, RRgRR)
            if error2_tmp < error2:
                M0, M1, error2, pinv_c = M0_tmp, M1_tmp, error2_tmp, c_off
    return (M0, M1), error2, pinv_c, eat_metric_error


def optimize_truncation(M0, M1, error2_old, fgf, fRR, fgRR, RRgRR, pinv_cutoffs, max_iter, tol_iter):
    r"""
    Optimizes the matrices M0 and M1 by minimizing
    truncation error using least square optimization.
    """
    gf = fgf.unfuse_legs(axes=0)
    gRR = fgRR.unfuse_legs(axes=0)

    for iter in range(1, max_iter+1):
        # fix M1 and optimize M0
        j0 = (gRR @ M1.H).fuse_legs(axes=[(0, 1)])
        g0 = tensordot(M1.conj(), gf, axes=(1, 1)).fuse_legs(axes=((1, 0), 2))
        g0 = g0.unfuse_legs(axes=1)
        g0 = tensordot(g0, M1, axes=(2, 1)).fuse_legs(axes=(0, (1, 2)))
        error_fun = lambda x: calculate_truncation_error2(x @ M1, fgf, fRR, RRgRR)
        M0, error2, pinv_c = optimal_pinv(g0, j0, pinv_cutoffs, error_fun)

        # fix M0 and optimize M1
        j1 = (M0.H @ gRR).fuse_legs(axes=[(0, 1)])
        g1 = tensordot(M0.conj(), gf, axes=(0, 0)).fuse_legs(axes=((0, 1), 2))
        g1 = g1.unfuse_legs(axes=1)
        g1 = tensordot(g1, M0, axes=(1, 0)).fuse_legs(axes=(0, (2, 1)))
        error_fun = lambda x: calculate_truncation_error2(M0 @ x, fgf, fRR, RRgRR)
        M1, error2, pinv_c = optimal_pinv(g1, j1, pinv_cutoffs, error_fun)

        # convergence condition
        if abs(error2 - error2_old) < tol_iter:
            break
        error2_old = error2

    return (M0, M1), error2, pinv_c, iter


def optimal_pinv(g, j, pinv_cutoffs, error_fun):
    r"""
    M = pinv(g) * j, where pinv cutoff is optimized based on error_fun.
    """
    S, U = g.eigh_with_truncation(axes=(0, 1), tol=min(pinv_cutoffs))
    smax = max(S._data)
    UHJ = U.H @ j
    error2, values_old = 100, len(S._data) + 1
    for cutoff in pinv_cutoffs:
        values = sum(S._data > cutoff * smax).item()
        if values < values_old:
            values_old = values
            Sr = S.reciprocal(cutoff=cutoff * smax)
            SrUHJ = Sr.broadcast(UHJ, axes=0)
            M_tmp = (U @ SrUHJ).unfuse_legs(axes=0)
            error2_tmp = error_fun(M_tmp)
            if error2_tmp < error2:
                M_new, error2, pinv_c = M_tmp, error2_tmp, cutoff
    return M_new, error2, pinv_c

def apply_predisentangler(env, bond, D_total, max_iter=400, tol=1e-7):

    diff = 0
    num_of_iter = 0

    psi = env.psi

    if isinstance(psi, Peps2Layers):
        psi = psi.ket  # to make it work with CtmEnv

    if bond is None:
        bonds = psi.bonds()
    else:
        bonds = [bond]

    for bond in bonds:
        dirn = psi.nn_bond_dirn(bond)
        s0, s1 = bond # if l_ordered else bond[::-1]

        tmpA = psi[s0]
        tmpB = psi[s1]

        if dirn == 'h':  # Horizontal gate, "lr" ordered

            Q0d, R0d = tmpA.qr(axes=((0, 1, 2), (3, 4)), sQ=-1)  # t l b rr @ rr r sa

            Q1d, R1d = tmpB.qr(axes=((0, 2, 3), (1, 4)), sQ=1, Qaxis=0, Raxis=-1)  # ll t b r @ l sa ll

            r0d = R0d.unfuse_legs(axes=2) # rr r s a
            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a

            r1d = R1d.unfuse_legs(axes=1) # l s' a' ll
            r1d = r1d.swap_gate(axes=(2, 3)) # swap_gate a' and ll

            r0d, r1d, diff, num_of_iter = predisentangler_iter(r0d, r1d, D_total, max_iter, tol)

            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a
            R0d = r0d.fuse_legs(axes=(0, 1, (2, 3)))

            r1d = r1d.swap_gate(axes=(2, 3))
            R1d = r1d.fuse_legs(axes=(0, (1, 2), 3))

            tmpA = tensordot(Q0d, R0d, axes=(3, 0)) # t l b r sa

            tmpB = tensordot(Q1d, R1d, axes=(0, 2)) # t b r l sa
            tmpB = tmpB.transpose(axes=(0, 3, 1, 2, 4))   # t l b r sa

        else: # dirn == 'v':  # Vertical gate, "tb" ordered

            Q0d, R0d = tmpA.qr(axes=((0, 1, 3), (2, 4)), sQ=-1)  # t l r bb @ bb b sa

            Q1d, R1d = tmpB.qr(axes=((1, 2, 3), (0, 4)), sQ=1, Qaxis=0, Raxis=-1)  # tt l b r @ t sa tt

            r0d = R0d.unfuse_legs(axes=2) # bb b s a
            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate b and a

            r1d = R1d.unfuse_legs(axes=1) # t s' a' tt
            r1d = r1d.swap_gate(axes=(2, 3)) # swap_gate a' and tt

            r0d, r1d, diff, num_of_iter = predisentangler_iter(r0d, r1d, D_total, max_iter, tol)

            r0d = r0d.swap_gate(axes=(1, 3)) # swap_gate r and a
            R0d = r0d.fuse_legs(axes=(0, 1, (2, 3)))

            r1d = r1d.swap_gate(axes=(2, 3))
            R1d = r1d.fuse_legs(axes=(0, (1, 2), 3))

            tmpA = tensordot(Q0d, R0d, axes=(3, 0)) # t l r b sa
            tmpA = tmpA.transpose(axes=(0, 1, 3, 2, 4))   # t l b r sa

            tmpB = tensordot(Q1d, R1d, axes=(0, 2)) # l b r t sa
            tmpB = tmpB.transpose(axes=(3, 0, 1, 2, 4))   # t l b r sa


        psi[s0] = tmpA
        psi[s1] = tmpB

    return diff, num_of_iter

def predisentangler_iter(r0d: Tensor, r1d: Tensor, D_total, max_iter, tol=1e-7):

    ii = 0
    diff = 32767

    r0dr1d = tensordot(r0d, r1d, axes=(1, 0)) # contr. r and l

    for ii in range(max_iter):

        u, s, v = svd_with_truncation(r0dr1d, axes=((0, 1, 2), (3, 4, 5)), sU=r0d.s[1], D_total=D_total)
        r0dr1d_new = s.broadcast(u, axes=3)
        r0dr1d_new = tensordot(r0dr1d_new, v, axes=(3, 0))

        g = build_predisentangler_g(r0dr1d, r0dr1d_new.conj()) # a a* a' a'*

        # xx s a s' a' yy
        r0dr1d_new = tensordot(r0dr1d, g, axes=((2, 4), (1, 3))) # xx s s' yy a a'
        r0dr1d_new = r0dr1d_new.transpose(axes=(0, 1, 4, 2, 5, 3))
        r0dr1d_new = r0dr1d_new / (r0dr1d_new.fuse_legs(axes=((0, 1, 2), (3, 4, 5))).norm(p="fro"))

        diff = (r0dr1d.fuse_legs(axes=((0, 1, 2), (3, 4, 5))) - r0dr1d_new.fuse_legs(axes=((0, 1, 2), (3, 4, 5)))).norm(p="fro") / r0dr1d.fuse_legs(axes=((0, 1, 2), (3, 4, 5))).norm(p="fro")

        r0dr1d = r0dr1d_new

        if diff < tol:
            u, s, v = svd_with_truncation(r0dr1d, axes=((0, 1, 2), (3, 4, 5)), sU=r0d.s[1], D_total=r0d.get_shape(axes=1))
            s = s.sqrt()
            r0d = s.broadcast(u, axes=3).transpose(axes=(0, 3, 1, 2))
            r1d = s.broadcast(v, axes=0)
            break

    return r0d, r1d, diff, ii

def build_predisentangler_g(r0dr1d: Tensor, r0dr1d_conj: Tensor):
    # r0dr1d xx s a s' a' yy
    Eg = tensordot(r0dr1d, r0dr1d_conj, axes=((0, 1, 3, 5), (0, 1, 3, 5))) # a a' a* a'*
    u, s, v = Eg.svd(axes=((0, 1), (2, 3)))
    Eg = tensordot(v.conj(), u.conj(), axes=(0, 2))
    Eg = Eg.transpose(axes=(0, 2, 1, 3))
    return Eg