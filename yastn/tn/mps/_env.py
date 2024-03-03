""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from itertools import groupby
from typing import Sequence
from . import MpsMpoOBC, MpoPBC, MpoTerm
from ._env_auxiliary import Env2, Env3_pbc, _Env_mpo_mpo_mpo, _Env_mps_mpo_mps, _Env_mpo_mpo_mpo_aux, _EnvParent
from ... import svd, qr, YastnError


def vdot(*args) -> number:
    r"""
    Calculate the overlap :math:`\langle \textrm{bra}|\textrm{ket}\rangle`,
    or :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle` depending on the number of provided agruments.

    Parameters
    -----------
    *args: yastn.tn.mps.MpsMpoOBC
    """
    if len(args) == 2:
        return measure_overlap(*args)
    return measure_mpo(*args)


def measure_overlap(bra, ket) -> number:
    r"""
    Calculate overlap :math:`\langle \textrm{bra}|\textrm{ket} \rangle`.
    Conjugate of MPS :code:`bra` is computed internally.

    MPSs :code:`bra` and :code:`ket` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra=bra, ket=ket)
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op: MpsMpoOBC | Sequence[tuple(MpsMpoOBC, number)], ket) -> number:
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.

    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    op: yastn.tn.mps.MpsMpoOBC or Sequence[tuple(MpsMpoOBC,number)]
        Operator written as (sums of) MPO.

    ket: yastn.tn.mps.MpsMpoOBC
    """
    env = Env(bra=bra, op=op, ket=ket)
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_1site(bra, O, ket) -> dict[int, number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i|\textrm{ket} \rangle` for local operator :code:`O` at each lattice site `i`.

    Local operators can be provided as dictionary {site: operator}, limiting the calculation to provided sites.
    Conjugate of MPS :code:`bra` is computed internally.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O: yastn.Tensor or dict
        An operator with signature (1, -1).
        It is possible to provide a dictionary {site: operator}

    ket: yastn.tn.mps.MpsMpoOBC
    """
    op = sorted(O.items()) if isinstance(O, dict) else [(n, O) for n in ket.sweep(to='last')]
    env = Env(bra=bra, ket=ket)
    env.setup_(to='first').setup_(to='last')
    results = {}
    for n, o in op:
        env.update_env_op_(n, o, to='first')
        results[n] = env.measure(bd=(n - 1, n))
    return results


def measure_2site(bra, O, P, ket, pairs=None) -> dict[tuple[int, int], number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i \textrm{P}_j|\textrm{ket} \rangle`
    of local operators :code:`O` and :code:`P` for each pair of lattice sites :math:`i < j`.

    Conjugate of MPS :code:`bra` is computed internally. Includes fermionic strings via swap_gate for fermionic operators.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpoOBC
        An MPS which will be conjugated.

    O, P: yastn.Tensor or dict
        Operators with signature (1, -1).
        It is possible to provide a dictionaries {site: operator}

    ket: yastn.tn.mps.MpsMpoOBC

    pairs: list[tuple[int, int]]
        It is possible to provide a list of pairs to limit the calculation.
        By default is None, when all pairs are calculated.
    """
    if pairs is not None:
        n1s = sorted(set(x[1] for x in pairs))
        pairs = sorted((-i, j) for i, j in pairs)
        pairs = [(-i, j) for i, j in pairs]
    else:
        n1s = range(ket.N)
        pairs = [(i, j) for i in range(ket.N - 1, -1, -1) for j in range(i + 1, ket.N)]

    env = Env(bra=bra, ket=ket)
    env.setup_(to='first').setup_(to='last')
    for n1 in n1s:
        env.update_env_op_(n1, P, to='first')

    results = {}
    for n0, n01s in groupby(pairs, key=lambda x: x[0]):
        env.update_env_op_(n0, O, to='last')
        _, n1 = next(n01s)
        for n in env.ket.sweep(to='last', df=n0 + 1):
            if n == n1:
                results[(n0, n1)] = env.measure(bd=(n - 1, n))
                try:
                    _, n1 = next(n01s)
                except StopIteration:
                    break
            env.update_env_(n, to='last')
    return dict(sorted(results.items()))


# The DMRG, TDVP operate over environments, which provide a way to compute
# * mat-vec product of effective H over interval (1-site, 2-site, ...),
#   where H is MPO or sum of MPOs


# def Env(bra=None, target=None, project=None):
#         if type(target) == MpsMpoOBC:  # mps
#             return Env2(bra=bra, ket=target)

#         # [mps]
#         # [mpo, mps]
#         # [mpo, mps]
#         # [MPS,]
#         # [MPO, MPS]
#         # [[MPO, MPO], MPS]

#         # [[MPS], [MPS],...]
#         # [[MPO, MPS], [MPO, MPS], ...]

#         # [[[MPO, MPO], MPS], [MPO, MPS], [MPS]]




#         if len(target) == 1:
#             env = Env(bra=psi, ket=target[0])
#         else:
#             env = Env(bra=psi, op=target[0], ket=target[1])

#         if op is None:
#             return Env2(bra=bra, ket=ket)
#         if type(op) == MpsMpoOBC:
#             if ket.nr_phys == 1:
#                 return _Env_mps_mpo_mps(bra, op, ket, project)
#             elif hasattr(op, 'mode') and op.mode == 'bra':
#                 _Env_mpo_mpo_mpo_aux(bra, op, ket, project)
#             else:
#                 return _Env_mpo_mpo_mpo(bra, op, ket, project)
#         elif type(op) == MpoPBC:
#             return Env3_pbc(bra, op, ket, project)
#         else:
#             return Env3_sum(bra, op, ket, project)

def Env(bra=None, op=None, ket=None, project=None):
        if op is None:
            return Env2(bra=bra, ket=ket)
        if type(op) == MpsMpoOBC:
            if ket.nr_phys == 1:
                return _Env_mps_mpo_mps(bra, op, ket, project)
            elif hasattr(op, 'mode') and op.mode == 'bra':
                _Env_mpo_mpo_mpo_aux(bra, op, ket, project)
            else:
                return _Env_mpo_mpo_mpo(bra, op, ket, project)
        elif type(op) == MpoPBC:
            return Env3_pbc(bra, op, ket, project)
        else:
            return Env3_sum(bra, op, ket, project)


class Env3_sum(_EnvParent):

    def __init__(self, bra=None, op:Sequence[MpoTerm]=None, ket=None, project=None):
        super().__init__(bra, project)
        envs, amps = [], []
        for o in [op] if type(op) == MpoTerm else op:
            if o.mpo.N != self.N:
                raise YastnError("all MPO operators and state should have the same number of sites")
            amps.append(o.amp)
            H = o.mpo
            if o.mode == 'bra':
                H = H.mpo.shallow_copy()
                H.mode = 'bra'
            envs.append(Env(bra, H, ket))
        self.envs = envs
        self.amps = amps

    def clear_site_(self, *args):
        for env in self.envs:
            env.clear_site_(*args)

    def factor(self):
        return 1

    def update_env_(self, n, to='last'):
        for env in self.envs:
            env.update_env_(n, to)
        if self.projector:
            self.projector._update_env(n, to)

    def measure(self, bd=(-1, 0)):
        return sum(amp * env.measure(bd) for amp, env in zip(self.amps, self.envs))

    def Heff0(self, C, bd):
        tmp = self.amps[0] * self.envs[0].Heff0(C, bd)
        for amp, env in zip(self.amps[1:], self.envs[1:]):
            tmp = tmp + amp * env.Heff0(C, bd)
        return tmp

    def Heff1(self, A, n):
        A = self._project_ort(A)
        tmp = self.amps[0] * self.envs[0].Heff1(A, n)
        for amp, env in zip(self.amps[1:], self.envs[1:]):
            tmp = tmp + amp * env.Heff1(A, n)
        return self._project_ort(tmp)

    def project_ket_on_bra_1(self, n):
        tmp = self.amps[0] * self.envs[0].project_ket_on_bra_1(n)
        for amp, env in zip(self.amps[1:], self.envs[1:]):
            tmp = tmp + amp * env.project_ket_on_bra_1(n)
        return self._project_ort(tmp)

    def Heff2(self, AA, bd):
        tmp = self._project_ort(AA)
        tmp = self.amps[0] * self.envs[0].Heff2(AA, bd)
        for amp, env in zip(self.amps[1:], self.envs[1:]):
            tmp = tmp + amp * env.Heff2(AA, bd)
        return self._project_ort(tmp)

    def project_ket_on_bra_2(self, bd):
        tmp = self.amps[0] * self.envs[0].project_ket_on_bra_2(bd)
        for amp, env in zip(self.amps[1:], self.envs[1:]):
            tmp = tmp + amp * env.project_ket_on_bra_2(bd)
        return self._project_ort(tmp)

    def enlarge_bond(self, bd, opts_svd):
        if bd[0] < 0 or bd[1] >= self.N:  # do not enlarge bond outside of the chain
            return False
        AL = self.bra[bd[0]]
        AR = self.bra[bd[1]]
        if any([(env.op[bd[0]].get_legs(axes=1).t != AL.get_legs(axes=1).t) or \
           (env.op[bd[1]].get_legs(axes=1).t != AR.get_legs(axes=1).t) for env in self.envs]):
            return True  # true if some charges are missing on physical legs of psi

        AL = AL.fuse_legs(axes=((0, 1), 2))
        AR = AR.fuse_legs(axes=(0, (1, 2)))
        shapeL = AL.get_shape()
        shapeR = AR.get_shape()
        if shapeL[0] == shapeL[1] or shapeR[0] == shapeR[1] or \
           ('D_total' in opts_svd and shapeL[1] >= opts_svd['D_total']):
            return False  # maximal bond dimension
        if 'tol' in opts_svd:
            _, R0 = qr(AL, axes=(0, 1), sQ=1)
            _, R1 = qr(AR, axes=(1, 0), Raxis=1, sQ=-1)
            S = svd(R0 @ R1, compute_uv=False)
            if any(S[t][-1] > opts_svd['tol'] * 1.1 for t in S.struct.t):
                return True
        return False
