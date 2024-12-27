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
""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from numbers import Number
from ... import eye, tensordot, ncon, vdot, YastnError, qr, svd
from . import MpsMpoOBC, MpoPBC
import abc
import copy


def Env(bra, target, **kwargs):
    r"""
    Initialize a proper environment supporting contraction of MPS/MPO's:
    :math:`\langle \textrm{bra} | \textrm{target} \rangle` where
    :math:`|\textrm{target} \rangle` can be an MPS/MPO, an operator acting on MPS/MPO, or a sum of thereof.

    Parameters
    ----------
    bra : yastn.tn.mps.MpsMpoOBC
        Can be an MPS or an MPO -- the target should be of the matching form.

    target : Sequence | yastn.tn.mps.MpsMpoOBC
        Dispatch over a set of supported targets:

        * ket or [ket] for :math:`\langle \textrm{bra} | \textrm{ket} \rangle`.
        * [mpo, ket]  for :math:`\langle \textrm{bra} | \textrm{mpo} | \textrm{ket} \rangle`.
        * [[mpo_1, mpo_2, ...], ket] for :math:`\langle \textrm{bra} | \sum_i \textrm{mpo}_i | \textrm{ket} \rangle`.
        * [[ket_1], [mpo_2, ket_2], [[mpo_3, mpo_4], ket_3]] for a sum of any combination of the above.

    Note
    ----
    :meth:`compression_<yastn.tn.mps.compression_>` directly calls :code:`Env(psi, target)`.
    :meth:`dmrg_<yastn.tn.mps.dmrg_>` and :meth:`tdvp_<yastn.tn.mps.tdvp_>` call :code:`Env(psi, target=[H, psi])`.
    """
    if isinstance(target, MpsMpoOBC):
        return Env2(bra=bra, ket=target)  # ket
    if isinstance(target[-1], MpsMpoOBC):
        ket = target[-1]
        if len(target) == 1:
            return Env2(bra=bra, ket=target[0])
        if len(target) == 2:
            op = target[0]
            if isinstance(op, MpsMpoOBC):
                if ket.nr_phys == 1:
                    if 'precompute' in kwargs and kwargs['precompute']:
                        return Env_mps_mpo_mps_precompute(bra, op, ket)
                    return Env_mps_mpo_mps(bra, op, ket)
                elif ket.nr_phys == 2 and hasattr(op, 'flag') and op.flag == 'on_bra':
                    return Env_mpo_mpobra_mpo(bra, op, ket)
                else:  # ket.nr_phys == 2 and default 'on_ket"
                    return Env_mpo_mpo_mpo(bra, op, ket)
            elif isinstance(op, MpoPBC):
                if bra.nr_phys != 1:
                    raise YastnError("Env: Application of MpoPBC on Mpo is not supported. Contact developers to add this functionality.")
                return Env_mps_mpopbc_mps(bra, op, ket)
            elif hasattr(op, '__iter__'):
                return Env_sum([Env(bra, [o, ket], **kwargs) for o in op])
            else:
                raise YastnError("Env: Input cannot be parsed.")
        if len(target) > 2:
            raise YastnError("Env: Input cannot be parsed.")

    # simple environments are handled
    # we flatten more complicated ones into Env_sum
    envs = []
    for tmp in target:
        if not hasattr(tmp, '__iter__'):
            raise YastnError("Env: Input cannot be parsed.")
        env = Env(bra, tmp, **kwargs)
        if isinstance(env, Env_sum):
            envs.extend(env.envs)
        else:
            envs.append(env)
    return Env_sum(envs)


class EnvParent(metaclass=abc.ABCMeta):

    def __init__(self, bra=None) -> None:
        """
        Interface for environments of 1D TNs.
        """
        self.config = bra.config
        self.bra = bra
        self.N = self.bra.N
        self.nr_phys = bra.nr_phys
        self.F = {}  # dict of envs dict[tuple[int, int], yastn.Tensor]

    def setup_(self, to='last'):
        r"""
        Setup all environments in given direction.

        Parameters
        ----------
        to: str
            :code:`first` or :code:`last`.
        """
        for n in self.bra.sweep(to=to):
            self.update_env_(n, to=to)
        return self

    def clear_site_(self, *args):
        r"""
        Clear environments pointing from sites whose indices are provided in args.
        """
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    @abc.abstractmethod
    def factor(self) -> Number:
        r"""
        Collect factors from constituent MPSs and MPOs.
        """

    @abc.abstractmethod
    def measure(self, bd=None) -> Number:
        r"""
        Calculate overlap between environments at :code:`bd` bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.
        """

    @abc.abstractmethod
    def update_env_(self, n, to='last'):
        r"""
        Update environment including site :code:`n`, in the direction given by :code:`to`.

        Parameters
        ----------
        n: int
            index of site to include to the environment

        to: str
            :code:`first` or :code:`last`.
        """

    @abc.abstractmethod
    def Heff0(self, C, bd) -> yastn.Tensor:
        r"""
        Action of Heff on central block :code:`Heff0 @ C`.

        Parameters
        ----------
        C: tensor
            a central block
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) as it is ordered]
        """

    @abc.abstractmethod
    def Heff1(self, A, n) -> yastn.Tensor:
        r"""
        Action of Heff on a single site MPS tensor :code:`Heff1 @ A`.

        Parameters
        ----------
        A: tensor
            site tensor

        n: int
            index of corresponding site
        """

    def project_ket_on_bra_1(self, n) -> yastn.Tensor:
        r"""
        Action of Heff1 on :code:`n`-th ket MPS tensor :code:`Heff1 @ ket[n]`.
        """
        return self.Heff1(self.ket[n], n) * self.ket.factor

    @abc.abstractmethod
    def Heff2(self, AA, bd) -> yastn.Tensor:
        r"""
        Action of Heff on central block :code:`Heff2 @ AA`.

        Parameters
        ----------
        AA: tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectively into 1-site update.
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) as it gets ordered]
        """

    def project_ket_on_bra_2(self, bd) -> yastn.Tensor:
        r"""
        Action of Heff2 on :code:`bd=(n, n+1)` ket MPS tensors :code:`Heff2 @ AA`.
        """
        return self.Heff2(self.ket.merge_two_sites(bd), bd) * self.ket.factor

    @abc.abstractmethod
    def charges_missing(self, n):
        """
        Auxiliary function for enlarge_bond.
        It checks if some charges are missing on the physical leg of the state
        as compared to the physical leg of the operator.

        In some corner cases, it is possible that symmetries might restrict
        1-site update from including this charge into the physical leg of psi.
        2-site update is better in such cases.
        """

    def enlarge_bond(self, bd, opts_svd):
        """
        Auxiliary function for site12 tdvp update.
        It decides if 1- or 2-site update should be used around a given bond.
        """
        if bd[0] < 0 or bd[1] >= self.N:
            return False  # do not enlarge bond outside of the chain

        if self.charges_missing(bd[0]) or self.charges_missing(bd[1]):
            return True  # true if some charges are missing on physical legs of psi

        indsL = (0, 1) if self.bra.nr_phys == 1 else (0, 1, 3)
        indsR = (1, 2) if self.bra.nr_phys == 1 else (1, 2, 3)
        AL = self.bra[bd[0]].fuse_legs(axes=(indsL, 2))
        AR = self.bra[bd[1]].fuse_legs(axes=(0, indsR))
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
                return True  # Schmidt values below expected tolerance

        return False  # no hint for using 2-site update

    def shallow_copy(self):
        r"""
        A shallow copy of environment class, that creates a new copy of dictionary storing env tensors.
        """
        env = copy.copy(self)
        env.F = dict(self.F)
        return env


class Env_sum(EnvParent):

    def __init__(self, envs):
        super().__init__(bra=envs[0].bra)
        self.envs=envs

    def clear_site_(self, *args):
        for env in self.envs:
            env.clear_site_(*args)

    def factor(self):
        return 1

    def update_env_(self, n, to='last'):
        for env in self.envs:
            env.update_env_(n, to)

    def measure(self, bd=(-1, 0)):
        return sum(env.measure(bd) for env in self.envs)

    def Heff0(self, C, bd):
        tmp = self.envs[0].Heff0(C, bd)
        for env in self.envs[1:]:
            tmp = tmp + env.Heff0(C, bd)
        return tmp

    def Heff1(self, A, n):
        tmp = self.envs[0].Heff1(A, n)
        for env in self.envs[1:]:
            tmp = tmp + env.Heff1(A, n)
        return tmp

    def project_ket_on_bra_1(self, n):
        tmp = self.envs[0].project_ket_on_bra_1(n)
        for env in self.envs[1:]:
            tmp = tmp + env.project_ket_on_bra_1(n)
        return tmp

    def Heff2(self, AA, bd):
        tmp = self.envs[0].Heff2(AA, bd)
        for env in self.envs[1:]:
            tmp = tmp + env.Heff2(AA, bd)
        return tmp

    def project_ket_on_bra_2(self, bd):
        tmp = self.envs[0].project_ket_on_bra_2(bd)
        for env in self.envs[1:]:
            tmp = tmp + env.project_ket_on_bra_2(bd)
        return tmp

    def charges_missing(self, n):
        return any(env.charges_missing(n) for env in self.envs)


class Env2(EnvParent):
    # The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.

    def __init__(self, bra=None, ket=None, n_left=None):
        super().__init__(bra)
        self.ket = ket

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('Env: bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YastnError('Env: bra and ket should have the same number of sites.')

        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj()]
        self.F[-1, 0] = eye(self.config, legs=legs, isdiag=False, n=n_left)
        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[self.N, self.N - 1] = eye(self.config, legs=legs, isdiag=False)

    def factor(self):
        return self.bra.factor * self.ket.factor

    def measure(self, bd=(-1, 0)):
        tmp = tensordot(self.F[bd], self.F[bd[::-1]], axes=((0, 1), (1, 0)))
        return self.factor() * tmp.to_number()

    def update_env_(self, n, to='last'):
        if to == 'first':
            tmp = tensordot(self.ket[n], self.F[n + 1, n], axes=(2, 0))
            tmp = tmp.swap_gate(axes=1, charge=self.F[n + 1, n].n)
            axes = ((1, 2), (1, 2)) if self.nr_phys == 1 else ((1, 3, 2), (1, 2, 3))
            self.F[n, n - 1] = tensordot(tmp, self.bra[n].conj(), axes=axes)
        else:  # to == 'last'
            tmp = tensordot(self.F[n - 1, n], self.ket[n], axes=((1, 0)))
            tmp = tmp.swap_gate(axes=1, charge=self.F[n - 1, n].n)
            axes = ((0, 1), (0, 1)) if self.nr_phys == 1 else ((0, 1, 3), (0, 1, 3))
            self.F[n, n + 1] = tensordot(self.bra[n].conj(), tmp, axes=axes)

    def Heff0(self, C, bd):
        raise NotImplementedError("Should not be triggered by current higher-level functions.")

    def Heff1(self, A, n):
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        return ncon([self.F[n - 1, n], A, self.F[n + 1, n]], inds)

    def Heff2(self, AA, bd):
        """ Heff2 @ AA """
        n1, n2 = bd
        axes = (0, (1, 2), 3) if AA.ndim == 4 else (0, (1, 2, 3, 5), 4)
        tmp = AA.fuse_legs(axes=axes)
        tmp = self.F[n1 - 1, n1] @ tmp @ self.F[n2 + 1, n2]
        tmp = tmp.unfuse_legs(axes=1)
        if tmp.ndim == 6:
            tmp = tmp.transpose(axes=(0, 1, 2, 3, 5, 4))
        return tmp

    def update_env_op_(self, n, op, to='first'):
        """
        Contractions for 2-layer environment update, with on-site operator ``op`` applied on site ``n``.

        If the operator has a charge, it gets propagated to the environment;
        In to='last', the charge of the environment is propagated with a proper swap gate,
        so that in measure_2site, a combination of to='last' and 'first'
        corresponds to the situation where the latter operator is applied first.
        Conventions are adapted to application in measure_1site and measure_2site,
        consistently with fermionic order.
        """
        if to == 'first':
            tmp = tensordot(self.ket[n], self.F[n + 1, n], axes=(2, 0))
            tmp = tensordot(op, tmp, axes=(1, 1))
            axes = ((0, 2), (1, 2)) if self.nr_phys == 1 else ((0, 3, 2), (1, 2, 3))
            self.F[n, n - 1] = tensordot(tmp, self.bra[n].conj(), axes=axes)
        else:  # to == 'last'
            tmp = tensordot(self.bra[n].conj(), self.F[n - 1, n], axes=((0, 0)))
            tmp = tensordot(op, tmp, axes=((0, 0)))
            tmp = tmp.swap_gate(axes=0, charge=tmp.n)
            axes = ((2, 0), (0, 1)) if self.nr_phys == 1 else ((3, 0, 2), (0, 1, 3))
            self.F[n, n + 1] = tensordot(tmp, self.ket[n], axes=axes)

    def charges_missing(self, n):
        raise NotImplementedError("Should not be triggered by current higher-level functions.")


class Env_project(Env2):
    def __init__(self, bra, proj, penalty=100):
        super().__init__(bra, proj)
        self.penalty = penalty
        assert self.nr_phys == 1, "Env_project works only for MPS, i.e., bra.nr_phys==1. Ask developers if extension needed."

    def Heff1(self, A, n):
        tmp = self.ket[n] @ self.F[n + 1, n]
        if (A.ndim == 2):
            tmp = tmp.fuse_legs(axes=(0, (1, 2)))
        tmp = self.F[n - 1, n] @ tmp
        return  tmp * (self.penalty * vdot(tmp, A))

    def Heff2(self, AA, bd):
        """ Heff2 @ AA """
        n1, n2 = bd
        tmp1 = self.F[n1 - 1, n1] @ self.ket[n1]
        tmp2 = self.ket[n2] @ self.F[n2 + 1, n2]
        if (AA.ndim == 2):
            tmp1 = tmp1.fuse_legs(axes=((0, 1), 2))
            tmp2 = tmp2.fuse_legs(axes=(0, (1, 2)))
        tmp = tmp1 @ tmp2
        return tmp * (self.penalty * vdot(tmp, AA))


class EnvParent_3(EnvParent):

    def __init__(self, bra, op, ket):
        super().__init__(bra)
        self.ket = ket
        self.op = op

        if op.N != self.N or ket.N != self.N:
            raise YastnError("Env: MPO operator, bra and ket should have the same number of sites.")
        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('Env: bra and ket should have the same number of physical legs.')
        if self.op.nr_phys != 2:
            raise YastnError('Env: MPO operator should have 2 physical legs.')

    def factor(self) -> float:
        return self.bra.factor * self.op.factor * self.ket.factor

    def charges_missing(self, n):
        op_t = self.op[n].get_legs(axes=1).t
        psi_t = self.bra[n].get_legs(axes=1).t
        return any(tt not in psi_t for tt in op_t)


class EnvParent_3_obc(EnvParent_3):

    def __init__(self, bra, op, ket):
        super().__init__(bra, op, ket)

        legs = [self.ket.virtual_leg('first').conj(), self.bra.virtual_leg('first')]
        legv = op.virtual_leg('first').conj()
        n_lt = ket.config.sym.add_charges(legv.t[0], signatures=(legv.s,), new_signature=-1)
        tmp = eye(self.config, legs=legs, isdiag=False, n=n_lt)
        self.F[-1, 0] = tmp.add_leg(axis=1, leg=legv)

        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        legv = op.virtual_leg('last').conj()
        n_rt = ket.config.sym.add_charges(legv.t[0], signatures=(legv.s,), new_signature=-1)
        tmp = eye(self.config, legs=legs, isdiag=False, n=n_rt)
        self.F[self.N, self.N - 1] = tmp.add_leg(axis=1, leg=legv)

    def measure(self, bd=(-1, 0)):
        return vdot(self.F[bd], self.F[bd[::-1]], conj=(0, 0)) * self.factor()

    def Heff0(self, C, bd):
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        tmp = tensordot(self.F[bd], C @ self.F[ibd], axes=((0, 1), (0, 1)))
        return tmp * self.op.factor


class Env_mps_mpo_mps(EnvParent_3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = self.F[n - 1, n] @ self.bra[n].conj()
            tmp = tensordot(self.op[n], tmp, axes=((0, 1), (1, 2)))
            self.F[n, n + 1] = tensordot(self.ket[n], tmp, axes=((0, 1), (2, 1)))
        elif to == 'first':
            tmp = self.ket[n] @ self.F[n + 1, n]
            tmp = tensordot(tmp, self.op[n], axes=((2, 1), (2, 3)))
            self.F[n, n - 1] = tensordot(tmp, self.bra[n].conj(), axes=((3, 1), (1, 2)))

    def Heff1(self, A, n):
        tmp = A @ self.F[n + 1, n]
        tmp = tensordot(self.op[n], tmp, axes=((2, 3), (2, 1)))
        tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1), (2, 0)))
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        tmp = AA @ self.F[n2 + 1, n2]
        tmp = tensordot(self.op[n2], tmp, axes=((2, 3), (3, 2)))
        tmp = tensordot(self.op[n1], tmp, axes=((2, 3), (0, 3)))
        tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (3, 0)))
        return tmp * self.op.factor

    def hole(self, n):
        """ Hole for peps tensor at site n. """
        tmp = self.F[n - 1, n] @ self.bra[n].conj()
        tmp = tensordot(tmp, self.F[n + 1, n], axes=(3, 2))
        tmp = tensordot(tmp, self.ket[n], axes=((3, 0), (2, 0)))
        return tmp.fuse_legs(axes=((0, 1), (2, 3)))


class Env_mps_mpo_mps_precompute(EnvParent_3_obc):

    def clear_site_(self, *args):
        r"""
        Clear environments pointing from sites whose indices are provided in args.
        """
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)
            self.F.pop((n, n - 1, n - 1), None)
            self.F.pop((n, n + 1, n + 1), None)

    def update_env_(self, n, to='last'):
        if to == 'last':
            if (n - 1, n, n) in self.F:
                Aket = self.ket[n].fuse_legs(axes=((0, 1), 2))
                Abra = self.bra[n].fuse_legs(axes=((0, 1), 2))
                tmp = tensordot(Aket, self.F[n - 1, n, n], axes=(0, 0))
                tmp = tmp @ Abra.conj()
            else:
                tmp = self.F[n - 1, n] @ self.bra[n].conj()
                tmp = tensordot(self.op[n], tmp, axes=((0, 1), (1, 2)))
                tmp = tensordot(self.ket[n], tmp, axes=((0, 1), (2, 1)))
            self.F[n, n + 1] = tmp
        elif to == 'first':
            if (n + 1, n, n) in self.F:
                Aket = self.ket[n].fuse_legs(axes=(0, (1, 2)))
                Abra = self.bra[n].fuse_legs(axes=(0, (1, 2)))
                tmp = Aket @ self.F[n + 1, n, n]
                tmp = tensordot(tmp, Abra.conj(), axes=(2, 1))
            else:
                tmp = self.ket[n] @ self.F[n + 1, n]
                tmp = tensordot(tmp, self.op[n], axes=((2, 1), (2, 3)))
                tmp = tensordot(tmp, self.bra[n].conj(), axes=((3, 1), (1, 2)))
            self.F[n, n - 1] = tmp

    def get_FL(self, n):
        if (n - 1, n, n) not in self.F:
            tmp = tensordot(self.F[n - 1, n], self.op[n], axes=(1, 0))
            self.F[n - 1, n, n] = tmp.fuse_legs(axes=((0, 4), 3, (1, 2)))
        return self.F[n - 1, n, n]

    def get_FR(self, n):
        if (n + 1, n, n) not in self.F:
            tmp = tensordot(self.op[n], self.F[n + 1, n], axes=(2, 1))
            self.F[n + 1, n, n] = tmp.fuse_legs(axes=((2, 3), 0, (1, 4)))
        return self.F[n + 1, n, n]

    def Heff1(self, A, n):
        FR = self.get_FR(n)
        tmp = tensordot(self.F[n - 1, n], A @ FR, axes=((0, 1), (0, 1)))
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        FL = self.get_FL(n1)
        FR = self.get_FR(n2)
        tmp = tensordot(FL, AA @ FR, axes=((0, 1), (0, 1)))
        return tmp * self.op.factor


class Env_mpo_mpo_mpo(EnvParent_3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = self.F[n - 1, n] @ self.bra[n].conj()
            tmp = tensordot(self.op[n], tmp, axes=((0, 1), (1, 2)))
            self.F[n, n + 1] = tensordot(self.ket[n], tmp, axes=((3, 0, 1), (4, 2, 1)))
        elif to == 'first':
            tmp = tensordot(self.ket[n], self.F[n + 1, n], axes=(2, 0))
            tmp = tensordot(tmp, self.op[n], axes=((3, 1), (2, 3)))
            self.F[n, n - 1] = tensordot(tmp, self.bra[n].conj(), axes=((4, 2, 1), (1, 2, 3)))

    def Heff1(self, A, n):
        tmp = tensordot(A, self.F[n + 1, n], axes=(2, 0))
        tmp = tensordot(self.op[n], tmp, axes=((2, 3), (3, 1)))
        tmp = tmp.transpose(axes=(2, 0, 1, 4, 3))
        tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1), (0, 1)))
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        tmp = tensordot(AA, self.F[n2 + 1, n2], axes=(4, 0))
        tmp = tensordot(self.op[n2], tmp, axes=((2, 3), (5, 3)))
        tmp = tensordot(self.op[n1], tmp, axes=((2, 3), (0, 3)))
        tmp = tmp.transpose(axes=(3, 0, 1, 4, 2, 6, 5))
        tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
        return tmp * self.op.factor


class Env_mpo_mpobra_mpo(EnvParent_3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = tensordot(self.F[n - 1, n], self.ket[n], axes=(0, 0))
            tmp = tensordot(tmp, self.op[n], axes=((0, 4), (0, 1)))
            self.F[n, n + 1] = tensordot(tmp, self.bra[n].conj(), axes=((0, 1, 4), (0, 1, 3)))
        elif to == 'first':
            tmp = tensordot(self.F[n + 1, n], self.bra[n].conj(), axes=(2, 2))
            tmp = tensordot(self.op[n], tmp, axes=((2, 3), (1, 4)))
            self.F[n, n - 1] = tensordot(self.ket[n], tmp, axes=((1, 2, 3), (4, 2, 1)))

    def Heff1(self, A, n):
        tmp = tensordot(A, self.F[n + 1, n], axes=(2, 0))
        tmp = tensordot(tmp, self.op[n], axes=((2, 3), (1, 2)))
        tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1), (0, 3)))
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        tmp = tensordot(AA, self.F[n2 + 1, n2], axes=(4, 0))
        tmp = tensordot(tmp, self.op[n2], axes=((4, 5), (1, 2)))
        tmp = tensordot(tmp, self.op[n1], axes=((2, 5), (1, 2)))
        tmp = tmp.transpose(axes=(0, 5, 1, 6, 2, 3, 4))
        tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1), (0, 1)))
        return tmp * self.op.factor

    def charges_missing(self, n):
        op_t = self.op[n].get_legs(axes=3).t
        psi_t = self.bra[n].get_legs(axes=3).t
        return any(tt not in psi_t for tt in op_t)


class EnvParent_3_pbc(EnvParent_3):

    def __init__(self, bra, op, ket):
        super().__init__(bra, op, ket)

        lfk = self.ket.virtual_leg('first')
        lfo = self.op.virtual_leg('first')
        lfb = self.bra.virtual_leg('first')
        tmp_oo = eye(self.config, legs=[lfo.conj(), lfo], isdiag=False)
        tmp_bk = eye(self.config, legs=[lfk.conj(), lfb], isdiag=False)
        self.F[-1, 0] = ncon([tmp_bk, tmp_oo], ((-0, -3), (-1, -2)))

        llk = self.ket.virtual_leg('last')
        llo = self.op.virtual_leg('last')
        llb = self.bra.virtual_leg('last')
        tmp_oo = eye(self.config, legs=[llo.conj(), llo],isdiag=False)
        tmp_bk = eye(self.config, legs=[llk.conj(), llb], isdiag=False)
        self.F[self.N, self.N - 1] = ncon([tmp_bk, tmp_oo], ((-0, -3), (-1, -2)))

    def measure(self, bd=(-1, 0)):
        return vdot(self.F[bd], self.F[bd[::-1]], conj=(0, 0)) * self.factor()

    def Heff0(self, C, bd):
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        tmp = tensordot(self.F[bd], C @ self.F[ibd], axes=((0, 1, 2), (0, 1, 2)))
        return tmp * self.op.factor


class Env_mps_mpopbc_mps(EnvParent_3_pbc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = self.F[n - 1, n] @ self.bra[n].conj()
            tmp = tensordot(self.op[n], tmp, axes=((0, 1), (1, 3)))
            self.F[n, n + 1] = tensordot(self.ket[n], tmp, axes=((0, 1), (2, 1)))
        elif to == 'first':
            tmp = tensordot(self.F[n + 1, n], self.bra[n].conj(), axes=(3, 2))
            tmp = tensordot(self.op[n], tmp, axes=((1, 2), (4, 1)))
            self.F[n, n - 1] = tensordot(self.ket[n], tmp, axes=((1, 2), (1, 2)))

    def Heff1(self, A, n):
        precompute = (A.ndim == 2)
        if precompute:
            A = A.unfuse_legs(axes=1)

        tmp = A @ self.F[n + 1, n]
        tmp = tensordot(self.op[n], tmp, axes=((2, 3), (2, 1)))
        tmp = tensordot(self.F[n - 1, n], tmp, axes=((0, 1, 2), (2, 0, 3)))

        if precompute:
            tmp = tmp.fuse_legs(axes=(0, (1, 2)))
        return tmp * self.op.factor

    def Heff2(self, AA, bd):
        precompute = (AA.ndim == 2)
        if precompute:
            AA = AA.unfuse_legs(axes=(0, 1))

        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        tmp = AA @ self.F[n2 + 1, n2]
        tmp = tensordot(self.op[n2], tmp, axes=((2, 3), (3, 2)))
        tmp = tensordot(self.op[n1], tmp, axes=((2, 3), (0, 3)))
        tmp = tensordot(self.F[n1 - 1, n1], tmp, axes=((0, 1, 2), (3, 0, 4)))

        if precompute:
            tmp = tmp.fuse_legs(axes=((0, 1), (2, 3)))
        return tmp * self.op.factor

    def hole(self, n):
        """ Hole for peps tensor at site n. """
        tmp = self.F[n - 1, n] @ self.bra[n].conj()
        tmp = tensordot(tmp, self.F[n + 1, n], axes=((2, 4), (2, 3)))
        tmp = tensordot(tmp, self.ket[n], axes=((3, 0), (2, 0)))
        return tmp.fuse_legs(axes=((0, 1), (2, 3)))
