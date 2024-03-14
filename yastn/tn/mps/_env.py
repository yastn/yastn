""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from ... import eye, tensordot, ncon, vdot, YastnError, qr, svd, ones
from . import MpsMpoOBC, MpoPBC
import abc


def Env(bra, target):
    """
    target is:
    ket  --  Env2
    [ket]  --  Env2
    [mpo, ket]  --  various Env3
    [[mpo, mpo], ket]  --  Env_sum
    [[ket], [mpo, ket], [[mpo, mpo], ket]]  --  Env_sum
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
                    return _Env_mps_mpo_mps(bra, op, ket)
                elif ket.nr_phys == 2 and hasattr(op, 'flag') and op.flag == 'on_bra':
                    return _Env_mpo_mpobra_mpo(bra, op, ket)
                else:  # ket.nr_phys == 2 and default 'on_ket"
                    return _Env_mpo_mpo_mpo(bra, op, ket)
            elif isinstance(op, MpoPBC):
                if bra.nr_phys != 1:
                    raise YastnError("Env: Application of MpoPBC on Mpo is not supported. Contact developers to add this functionality.")
                return _Env_mps_mpopbc_mps(bra, op, ket)
            elif hasattr(op, '__iter__'):
                return Env_sum([Env(bra, [o, ket]) for o in op])
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
        env = Env(bra, tmp)
        if isinstance(env, Env_sum):
            envs.extend(env.envs)
        else:
            envs.append(env)
    return Env_sum(envs)


class _EnvParent(metaclass=abc.ABCMeta):

    def __init__(self, bra=None) -> None:
        """
        Interface for environments of 1D TNs. In particular of the form,

            <bra| (sum_i op_i |ket_i>)

        where op_i can be None/identity.
        """
        self.config = bra.config
        self.bra = bra
        self.N = self.bra.N
        self.nr_phys = bra.nr_phys
        self.F = {}  # dict of envs dict[tuple[int, int], yastn.Tensor]

    def setup_(self, to='last'):
        r"""
        Setup all environments in the direction given by ``to``.

        Parameters
        ----------
        to: str
            'first' or 'last'.
        """
        for n in self.bra.sweep(to=to):
            self.update_env_(n, to=to)
        return self

    def clear_site_(self, *args):
        r"""
        Clear environments pointing from sites which indices are provided in args.
        """
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    @abc.abstractmethod
    def factor(self) -> number:
        r"""
        Collect factors from constituent MPSs and MPOs.
        """

    @abc.abstractmethod
    def measure(self, bd=None) -> number:
        r"""
        Calculate overlap between environments at bd bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.
        """

    @abc.abstractmethod
    def update_env_(self, n, to='last'):
        r"""
        Update environment including site n, in the direction given by to.

        Parameters
        ----------
        n: int
            index of site to include to the environment

        to: str
            'first' or 'last'.
        """

    @abc.abstractmethod
    def Heff0(self, C, bd) -> yastn.Tensor:
        r"""
        Action of Heff on central block, Heff0 @ C

        Parameters
        ----------
        C: tensor
            a central block
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]
        """

    @abc.abstractmethod
    def Heff1(self, A, n) -> yastn.Tensor:
        r"""
        Action of Heff on a single site MPS tensor, Heff1 @ A

        Parameters
        ----------
        A: tensor
            site tensor

        n: int
            index of corresponding site
        """

    def project_ket_on_bra_1(self, n) -> yastn.Tensor:
        r"""
        Action of Heff1 on n-th ket MPS tensor, Heff1 @ ket[n]
        """
        return self.Heff1(self.ket[n], n)

    @abc.abstractmethod
    def Heff2(self, AA, bd) -> yastn.Tensor:
        r"""
        Action of Heff on central block, Heff2 @ AA.

        Parameters
        ----------
        AA: tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectivly into 1-site update.
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]
        """

    def project_ket_on_bra_2(self, bd) -> yastn.Tensor:
        r"""
        Action of Heff2 on bd = (n, n+1) ket MPS tensors, Heff2 @ AA
        """
        return self.Heff2(self.ket.merge_two_sites(bd), bd)

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

        AL = self.bra[bd[0]].fuse_legs(axes=((0, 1), 2))
        AR = self.bra[bd[1]].fuse_legs(axes=(0, (1, 2)))
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


class Env_sum(_EnvParent):

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


class Env2(_EnvParent):
    # The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.

    def __init__(self, bra=None, ket=None):
        super().__init__(bra)
        self.ket = ket

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('Env: bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YastnError('Env: bra and ket should have the same number of sites.')

        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj()]
        self.F[(-1, 0)] = eye(self.config, legs=legs, isdiag=False)
        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[(self.N, self.N - 1)] = eye(self.config, legs=legs, isdiag=False)

    def factor(self):
        return self.bra.factor * self.ket.factor

    def measure(self, bd=(-1, 0)):
        tmp = tensordot(self.F[bd], self.F[bd[::-1]], axes=((0, 1), (1, 0)))
        return self.factor() * tmp.to_number()

    def update_env_(self, n, to='last'):
        if to == 'first':
            inds = ((-0, 2, 1), (1, 3), (-1, 2, 3)) if self.nr_phys == 1 else ((-0, 2, 1, 4), (1, 3), (-1, 2, 3, 4))
            self.F[(n, n - 1)] = ncon([self.ket[n], self.F[(n + 1, n)], self.bra[n].conj()], inds)
        elif to == 'last':
            inds = ((2, 3, -0), (2, 1), (1, 3, -1)) if self.nr_phys == 1 else ((2, 3, -0, 4), (2, 1), (1, 3, -1, 4))
            self.F[(n, n + 1)] = ncon([self.bra[n].conj(), self.F[(n - 1, n)], self.ket[n]], inds)

    def Heff0(self, C, bd):
        raise YastnError("Should not be triggered by current higher-level functions.")  # pragma: no cover
        # bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        # C = self.op.factor * C
        # return self.F[bd] @ C @ self.F[ibd]

    def Heff1(self, x, n):
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        return ncon([self.F[(n - 1, n)], x, self.F[(n + 1, n)]], inds)

    def Heff2(self, AA, bd):
        """ Heff2 @ AA """
        n1, n2 = bd
        axes = (0, (1, 2), 3) if AA.ndim == 4 else (0, (1, 2, 3, 5), 4)
        temp = AA.fuse_legs(axes=axes)
        temp = self.F[(n1 - 1, n1)] @ temp @ self.F[(n2 + 1, n2)]
        temp = temp.unfuse_legs(axes=1)
        if temp.ndim == 6:
            temp = temp.transpose(axes=(0, 1, 2, 3, 5, 4))
        return temp

    def update_env_op_(self, n, op, to='first'):
        """
        Contractions for 2-layer environment update, with on-site operator ``op`` applied on site ``n``.
        """
        if to == 'first':
            temp = tensordot(self.ket[n], self.F[(n + 1, n)], axes=(2, 0))
            op = op.add_leg(axis=0, s=1)
            temp = tensordot(op, temp, axes=(2, 1))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            axes = ((0, 2), (1, 2)) if self.nr_phys == 1 else ((0, 3, 2), (1, 2, 3))
            self.F[(n, n - 1)] = tensordot(temp, self.bra[n].conj(), axes=axes)
        else:  # to == 'last'
            op = op.add_leg(axis=0, s=1)
            temp = tensordot(op, self.ket[n], axes=((2, 1)))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            temp = tensordot(self.F[(n - 1, n)], temp, axes=((1, 1)))
            axes = ((0, 1), (0, 1)) if self.nr_phys == 1 else ((0, 1, 3), (0, 1, 3))
            self.F[(n, n + 1)] = tensordot(self.bra[n].conj(), temp, axes=axes)

    def charges_missing(self, n):
        raise YastnError("Should not be triggered by current higher-level functions.")  # pragma: no cover


class Env_project(Env2):
    def __init__(self, bra, proj, penalty=100):
        super().__init__(bra, proj)
        self.penalty = penalty

    def Heff1(self, x, n):
        pp = super().Heff1(self.ket[n], n)
        return self.penalty * pp * vdot(pp, x)

    def Heff2(self, AA, bd):
        """ Heff2 @ AA """
        pp = self.ket.merge_two_sites(bd)
        pp = super().Heff2(pp, bd)
        return (self.penalty * vdot(pp, AA)) * pp

class _EnvParent_3(_EnvParent):

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



class _EnvParent_3_obc(_EnvParent_3):

    def __init__(self, bra, op, ket):
        super().__init__(bra, op, ket)

        # init boundaries
        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj()]
        tmp = eye(self.config, legs=legs, isdiag=False)
        self.F[(-1, 0)] = tmp.add_leg(axis=1, leg=op.virtual_leg('first').conj())

        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        tmp = eye(self.config, legs=legs, isdiag=False)
        self.F[(self.N, self.N - 1)] = tmp.add_leg(axis=1, leg=op.virtual_leg('last').conj())

    def measure(self, bd=(-1, 0)):
        tmp = tensordot(self.F[bd], self.F[bd[::-1]], axes=((0, 1, 2), (2, 1, 0)))
        return self.factor() * tmp.to_number()

    def Heff0(self, C, bd):
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        C = self.op.factor * C
        tmp = self.F[bd].tensordot(C, axes=(2, 0))
        return tmp.tensordot(self.F[ibd], axes=((1, 2), (1, 0)))

class _Env_mps_mpo_mps(_EnvParent_3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = ncon([self.bra[n].conj(), self.F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
            tmp = self.op[n]._attach_01(tmp)
            self.F[(n, n + 1)] = ncon([tmp, self.ket[n]], ((-0, -1, 1, 2), (1, 2, -2)))
        elif to == 'first':
            tmp = self.ket[n] @ self.F[(n + 1, n)]
            tmp = self.op[n]._attach_23(tmp)
            self.F[(n, n - 1)] = ncon([tmp, self.bra[n].conj()], ((-0, -1, 1, 2), (-2, 2, 1)))

    def Heff1(self, A, n):
        nl, nr = n - 1, n + 1
        tmp = A @ self.F[(nr, n)]
        tmp = self.op[n]._attach_23(tmp)
        tmp = ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
        return self.op.factor * tmp

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1
        tmp = AA.fuse_legs(axes=((0, 1), 2, 3))
        tmp = tmp @ self.F[(nr, n2)]
        tmp = self.op[n2]._attach_23(tmp)
        tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
        tmp = tmp.unfuse_legs(axes=0)
        tmp = self.op[n1]._attach_23(tmp)
        tmp = ncon([self.F[(nl, n1)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
        tmp = tmp.unfuse_legs(axes=2)
        return self.op.factor * tmp


class _Env_mpo_mpo_mpo(_EnvParent_3_obc):

    def update_env_(self, n, to='last'):
        if to == 'last':
            bA = self.bra[n].fuse_legs(axes=(0, 1, (2, 3)))
            tmp = ncon([bA.conj(), self.F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
            tmp = self.op[n]._attach_01(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            self.F[(n, n + 1)] = ncon([tmp, self.ket[n]], ((-0, 3, -1, 1, 2), (1, 2, -2, 3)))
        elif to == 'first':
            kA = self.ket[n].fuse_legs(axes=((0, 3), 1, 2))
            tmp = ncon([kA, self.F[(n + 1, n)]], ((-0, -1, 1), (1, -2, -3)))
            tmp = self.op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            self.F[(n, n - 1)] = ncon([tmp, self.bra[n].conj()], ((-0, 3, -1, 1, 2), (-2, 2, 1, 3)))

    def Heff1(self, A, n):
        nl, nr = n - 1, n + 1
        tmp = A.fuse_legs(axes=((0, 3), 1, 2))
        tmp = tmp @ self.F[(nr, n)]
        tmp = self.op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        tmp = ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, -3, 1, -2, -1)))
        return self.op.factor * tmp

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1
        tmp = AA.fuse_legs(axes=((0, 2, 5), 1, 3, 4))
        tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
        tmp = tmp @ self.F[(nr, n2)]
        tmp = self.op[n2]._attach_23(tmp)
        tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
        tmp = tmp.unfuse_legs(axes=0)
        tmp = self.op[n1]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        tmp = ncon([self.F[(nl, n1)], tmp], ((-0, 1, 2), (2, -2, -4, 1, -3, -1)))
        tmp = tmp.unfuse_legs(axes=3)
        return self.op.factor * tmp


class _Env_mpo_mpobra_mpo(_EnvParent_3_obc):
    def update_env_(self, n, to='last'):
        if to == 'last':
            tmp = ncon([self.ket[n], self.F[(n - 1, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
            tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
            tmp = self.op[n]._attach_01(tmp)
            bA = self.bra[n].fuse_legs(axes=((0, 1), 2, 3))
            self.F[(n, n + 1)] = ncon([bA.conj(), tmp], ((1, -0, 2), (-2, -1, 1, 2)))
        elif to == 'first':
            bA = self.bra[n].fuse_legs(axes=((0, 1), 2, 3))
            tmp = ncon([bA.conj(), self.F[(n + 1, n)]], ((-0, 1, -1), (-3, -2, 1)))
            tmp = self.op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            self.F[(n, n - 1)] = ncon([self.ket[n], tmp], ((-0, 1, 2, 3), (-2, 1, -1, 2, 3)))

    def Heff1(self, A, n):
        nl, nr = n - 1, n + 1
        tmp = A.fuse_legs(axes=(0, (1, 2), 3))
        tmp = ncon([tmp, self.F[(nl, n)]], ((1, -0, -1), (-3, -2, 1)))
        tmp = self.op[n]._attach_01(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        tmp = ncon([tmp, self.F[(nr, n)]], ((-1, 1, 2, -0, -3), (1, 2, -2)))
        return self.op.factor * tmp

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1
        tmp = AA.fuse_legs(axes=(0, 2, (1, 3, 4), 5))
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        tmp = ncon([tmp, self.F[(nl, n1)]], ((1, -1, -0), (-3, -2, 1)))
        tmp = self.op[n1]._attach_01(tmp)
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        tmp = tmp.unfuse_legs(axes=0)
        tmp = self.op[n2]._attach_01(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        tmp = ncon([tmp, self.F[(nr, n2)]], ((-1, -2, 1, 2, -0, -4), (1, 2, -3)))
        tmp = tmp.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3, 4, 5))
        return self.op.factor * tmp

    def charges_missing(self, n):
        op_t = self.op[n].get_legs(axes=3).t
        psi_t = self.bra[n].get_legs(axes=3).t
        return any(tt not in psi_t for tt in op_t)


class _EnvParent_3_pbc(_EnvParent_3):

    def __init__(self, bra, op, ket):
        super().__init__(bra, op, ket)

        # left boundary
        lfb = self.bra.virtual_leg('first')
        lfo = self.op.virtual_leg('first')
        lfk = self.ket.virtual_leg('first')
        tmp_oo = eye(self.config, legs=lfo.conj(), isdiag=False)
        tmp_bk = eye(self.config, legs=[lfb, lfk.conj()], isdiag=False)
        self.F[(-1, 0)] = ncon([tmp_oo, tmp_bk], ((-1, -2), (-0, -3)))

        # right boundary
        llk = self.ket.virtual_leg('last')
        llo = self.op.virtual_leg('last')
        llb = self.bra.virtual_leg('last')
        tmp_oo = eye(self.config, legs=llo.conj(), isdiag=False)
        tmp_bk = eye(self.config, legs=[llk.conj(), llb], isdiag=False)
        self.F[(self.N, self.N - 1)] = ncon([tmp_oo, tmp_bk], ((-1, -2), (-0, -3)))

    def Heff0(self, C, bd):
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        C = self.op.factor * C
        tmp = self.F[bd].tensordot(C, axes=(3, 0))
        return tmp.tensordot(self.F[ibd], axes=((3, 1, 2), (0, 1, 2)))

    def measure(self, bd=(-1, 0)):
        axes = ((0, 1, 2, 3), (3, 1, 2, 0))
        return self.factor() * self.F[bd].tensordot(self.F[bd[::-1]], axes=axes).to_number()


class _Env_mps_mpopbc_mps(_EnvParent_3_pbc):

    def __init__(self, bra=None, op=None, ket=None):
        super().__init__(bra, op, ket)

    def Heff1(self, A, n):
        nl, nr = n - 1, n + 1
        Fr = self.F[(nr, n)].fuse_legs(axes=(0, 1, (2, 3)))
        tmp = A.tensordot(Fr, axes=(2, 0))
        tmp = self.op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=2)
        tmp = self.F[(nl, n)].tensordot(tmp, axes=((3, 1, 2), (0, 1, 2)))
        tmp = tmp.transpose(axes=(0, 2, 1))
        return self.op.factor * tmp

    def Heff2(self, AA, bd):
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1
        Fr = self.F[(nr, n2)].fuse_legs(axes=(0, 1, (2, 3)))
        tmp = AA.fuse_legs(axes=((0, 1), 2, 3))
        tmp = tmp.tensordot(Fr, axes=(2, 0))
        tmp = self.op[n2]._attach_23(tmp)
        tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        tmp = tmp.unfuse_legs(axes=0)
        tmp = self.op[n1]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=2)
        tmp = tmp.unfuse_legs(axes=2)
        tmp = self.F[(nl, n1)].tensordot(tmp, axes=((3, 1, 2), (0, 1, 2)))
        tmp = tmp.transpose(axes=(0, 3, 2, 1))
        return self.op.factor * tmp

    def hole(self, n):
        """ Hole for peps tensor at site n. """
        nl, nr = n - 1, n + 1
        tmp = self.F[(nl, n)].tensordot(self.ket[n], axes=(3, 0))
        tmp = tmp.tensordot(self.F[(nr, n)], axes=((2, 4), (2, 0)))
        tmp = tmp.tensordot(self.bra[n].conj(), axes=((0, 4), (0, 2)))
        return tmp.transpose(axes=(0, 3, 2, 1))

    def update_env_(self, n, to='last'):
        if to == 'last':
            bran = self.bra[n].transpose(axes=(2, 1, 0)).conj()
            tmp = self.F[(n - 1, n)].fuse_legs(axes=(0, 1, (2, 3)))
            tmp = bran.tensordot(tmp, axes=(2, 0))
            tmp = self.op[n]._attach_01(tmp)
            tmp = tmp.unfuse_legs(axes=2)
            self.F[(n, n + 1)] = tmp.tensordot(self.ket[n], axes=((3, 4), (0, 1)))
        elif to == 'first':
            tmp = self.F[(n + 1, n)].fuse_legs(axes=(0, 1, (2, 3)))
            tmp = self.ket[n].tensordot(tmp, axes=(2, 0))
            tmp = self.op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=2)
            self.F[(n, n - 1)] = tmp.tensordot(self.bra[n].conj(), axes=((3, 4), (2, 1)))


# class _Env_mpo_mpopbc_mpo  _Env_mpo_mpopbcbra_mpo
    # def Heff1(self, A, n):
        # elif self.nr_phys == 2 and not self.on_aux:
        #     tmp = tmp.fuse_legs(axes=((0, 3), 1, 2))
        #     Fr = self.F[(nr, n)].fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = tmp.tensordot(Fr, axes=(2, 0))
        #     tmp = self.op[n]._attach_23(tmp)
        #     tmp = tmp.unfuse_legs(axes=(0, 2))
        #     tmp = tmp.swap_gate(axes=(1, 3))
        #     tmp = self.F[(nl, n)].tensordot(tmp, axes=((1, 2, 3), (2, 3, 0)))
        #     tmp = tmp.transpose(axes=(0, 3, 2, 1))
        # else:  # if self.nr_phys == 2 and self.on_aux:    #todo
        #     tmp = tmp.fuse_legs(axes=(0, (1, 2), 3))
        #     tmp = ncon([tmp, self.F[(nl, n)]], ((1, -0, -1), (-3, -2, 1)))
        #     tmp = self.op[n]._attach_01(tmp)
        #     tmp = tmp.unfuse_legs(axes=0)
        #     tmp = ncon([tmp, self.F[(nr, n)]], ((-1, 1, -0, 2, -3), (1, 2, -2)))

    # def Heff2(self, AA, bd):
        # if self.nr_phys == 2 and not self.on_aux:  # for mpo_mpopbc_mpo
        #     Fr = self.F[(nr, n2)].fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = tmp.fuse_legs(axes=(0, 1, (2, 5), 3, 4))
        #     tmp = tmp.fuse_legs(axes=((0, 2), 1, 3, 4))
        #     tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
        #     tmp = tmp.tensordot(Fr, axes=(2, 0))
        #     tmp = self.op[n2]._attach_23(tmp)
        #     tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = tmp.unfuse_legs(axes=0)
        #     tmp = self.op[n1]._attach_23(tmp)
        #     tmp = tmp.unfuse_legs(axes=2)
        #     tmp = tmp.unfuse_legs(axes=(0, 2))
        #     tmp = tmp.swap_gate(axes=(1, 3))
        #     tmp = self.F[(nl, n1)].tensordot(tmp, axes=((3, 1, 2), (0, 2, 3)))
        #     tmp = tmp.unfuse_legs(axes=1)
        #     tmp = tmp.transpose(axes=(0, 5, 1, 4, 3, 2))
        # else:  # if self.nr_phys == 2 and self.on_aux:  todo
        #     tmp = tmp.fuse_legs(axes=(0, 2, (1, 3, 4), 5))
        #     tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = ncon([tmp, self.F[(nl, n1)]], ((1, -1, -0), (-3, -2, 1)))
        #     tmp = self.op[n1]._attach_01(tmp)
        #     tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = tmp.unfuse_legs(axes=0)
        #     tmp = self.op[n2]._attach_01(tmp)
        #     tmp = tmp.unfuse_legs(axes=0)
        #     tmp = ncon([tmp, self.F[(nr, n2)]], ((-1, -2, 1, 2, -0, -4), (1, 2, -3)))
        #     tmp = tmp.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3, 4, 5))

    # def update_env_(self, n, to='last'):
        # if nr_phys == 2 and not on_aux and to == 'last':
        #     bran = bra[n].fuse_legs(axes=((2, 3), 1, 0)).conj()
        #     tmp = F[(n - 1, n)].fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = bran.tensordot(tmp, axes=(2, 0))
        #     tmp = op[n]._attach_01(tmp)
        #     tmp = tmp.unfuse_legs(axes=(0, 2))
        #     tmp = tmp.swap_gate(axes=(1, 3))
        #     F[(n, n + 1)] = tmp.tensordot(ket[n], axes=((4, 5, 1), (0, 1, 3)))
        # elif nr_phys == 2 and not on_aux and to == 'first':
        #     ketn = ket[n].fuse_legs(axes=((0, 3), 1, 2))
        #     tmp = F[(n + 1, n)].fuse_legs(axes=(0, 1, (2, 3)))
        #     tmp = ketn.tensordot(tmp, axes=(2, 0))
        #     tmp = op[n]._attach_23(tmp)
        #     tmp = tmp.unfuse_legs(axes=(0, 2))
        #     tmp = tmp.swap_gate(axes=(1, 3))
        #     F[(n, n - 1)] = tmp.tensordot(bra[n].conj(), axes=((1, 4, 5), (3, 2, 1)))
        # elif nr_phys == 2 and on_aux and to == 'last':  # todo
        #     tmp = ncon([ket[n], F[(n - 1, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
        #     tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
        #     tmp = op[n]._attach_01(tmp)
        #     bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        #     F[(n, n + 1)] = ncon([bA.conj(), tmp], ((1, -0, 2), (-2, -1, 1, 2)))
        # else: # nr_phys == 2 and on_aux and to == 'first':  # todo
        #     bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        #     tmp = ncon([bA.conj(), F[(n + 1, n)]], ((-0, 1, -1), (-3, -2, 1)))
        #     tmp = op[n]._attach_23(tmp)
        #     tmp = tmp.unfuse_legs(axes=0)
        #     F[(n, n - 1)] = ncon([ket[n], tmp], ((-0, 1, 2, 3), (-2, 1, -1, 2, 3)))
