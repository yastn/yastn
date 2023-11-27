""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from ... import tensor, initialize, YastnError, expmv
from itertools import groupby
from typing import Sequence, Dict, Optional
from ...tensor import Tensor
from . import MpsMpo
import abc
from math import prod

def vdot(*args) -> number:
    r"""
    Calculate the overlap :math:`\langle \textrm{bra}|\textrm{ket}\rangle`,
    or :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle` depending on the number of provided agruments.

    Parameters
    -----------
    *args: yastn.tn.mps.MpsMpo
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
    bra: yastn.tn.mps.MpsMpo
        An MPS which will be conjugated.

    ket: yastn.tn.mps.MpsMpo
    """
    env = Env2(bra=bra, ket=ket)
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op: MpsMpo | Sequence[tuple(MpsMpo,number)], ket) -> number:
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.

    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpo
        An MPS which will be conjugated.

    op: yastn.tn.mps.MpsMpo or Sequence[tuple(MpsMpo,number)]
        Operator written as (sums of) MPO.

    ket: yastn.tn.mps.MpsMpo
    """
    env = Env3(bra=bra, op=op, ket=ket)
    env.setup_(to='first')
    return env.measure(bd=(-1, 0))


def measure_1site(bra, O, ket) -> dict[int, number]:
    r"""
    Calculate expectation values :math:`\langle \textrm{bra}|\textrm{O}_i|\textrm{ket} \rangle` for local operator :code:`O` at each lattice site `i`.

    Local operators can be provided as dictionary {site: operator}, limiting the calculation to provided sites.
    Conjugate of MPS :code:`bra` is computed internally.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpo
        An MPS which will be conjugated.

    O: yastn.Tensor or dict
        An operator with signature (1, -1).
        It is possible to provide a dictionary {site: operator}

    ket: yastn.tn.mps.MpsMpo
    """
    op = sorted(O.items()) if isinstance(O, dict) else [(n, O) for n in ket.sweep(to='last')]
    env = Env2(bra=bra, ket=ket)
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
    bra: yastn.tn.mps.MpsMpo
        An MPS which will be conjugated.

    O, P: yastn.Tensor or dict
        Operators with signature (1, -1).
        It is possible to provide a dictionaries {site: operator}

    ket: yastn.tn.mps.MpsMpo

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

    env = Env2(bra=bra, ket=ket)
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


class _EnvParent(metaclass=abc.ABCMeta):

    def __init__(self, bra=None, ket=None, project=None) -> None:
        """
        Holds environments of 1D TNs. In particular of type

              <bra| B_0--B_1--...(i-1,i)--B_i--(i,i+1)...--B_N-1--B_N
                    
        \sum_i c_i*(O_0--O_1--...(i-1,i)--O_i--(i,i+1)...--O_N-1--O_N)_i  

                    K_0--K_1--...(i-1,i)--K_i--(i,i+1)...--K_N-1--K_N |ket>

        with optional sum of MPOs. To retrieve specific environments ...

        """
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.N = ket.N
        self.nr_phys = ket.nr_phys
        # environments, indexed by bonds with respect to i-th MPO 
        self.F : Sequence[Dict[tuple(int),Tensor]]= []
        self.ort = [] if project is None else project
        # environments, indexed by bonds with respect to k-th MPS-based projector
        self.Fort : Sequence[Dict[tuple(int),Tensor]] = [{} for _ in range(len(self.ort))]
        self._temp = {}
        self.reset_temp_()

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('MpsMpo for bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YastnError('MpsMpo for bra and ket should have the same number of sites.')

        config = self.ket[0].config
        # initialize environments with respect to orthogonal projections
        for ii in range(len(self.ort)):
            legs = [self.ort[ii].virtual_leg('first'), self.ket.virtual_leg('first').conj()]
            self.Fort[ii][(-1, 0)] = initialize.ones(config=config, legs=legs)
            legs = [self.ket.virtual_leg('last').conj(), self.ort[ii].virtual_leg('last')]
            self.Fort[ii][(self.N, self.N - 1)] = initialize.ones(config=config, legs=legs)

    # def __getitem__(self, key : tuple(int) | tuple(int,tuple(int))):
    #     if len(self.F)==1:
    #         return self.F[0][key]
    #     return self.F[key[0]][key[1]]

    # def __setitem__(self, key : tuple(int) | tuple(int,tuple(int)), item):
    #     if len(self.F)==1:
    #         self.F[0][key]= item
    #     else:
    #         self.F[key[0]][key[1]]= item

    def reset_temp_(self):
        """ Reset temporary objects stored to speed-up some simulations. """
        self._temp = {'Aort': [], 'op_2site': {}, 'expmv_ncv': {}}

    def setup_(self, to='last'):
        r"""
        Setup all environments in the direction given by ``to``.

        Parameters
        ----------
        to: str
            'first' or 'last'.
        """
        for n in self.ket.sweep(to=to):
            self.update_env_(n, to=to)
        return self

    def clear_site_(self, *args):
        r""" Clear environments pointing from sites which indices are provided in args. """
        for _f in self.F:
            for n in args:
                _f.pop((n, n - 1), None)
                _f.pop((n, n + 1), None)

    @abc.abstractmethod
    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at bd bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.

        Returns
        -------
        overlap: float or complex
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

    def update_Aort_(self, n):
        """ Update projection of states to be subtracted from psi. """
        Aort = []
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        for ii in range(len(self.ort)):
            Aort.append(tensor.ncon([self.Fort[ii][(n - 1, n)], self.ort[ii][n], self.Fort[ii][(n + 1, n)]], inds))
        self._temp['Aort'] = Aort

    def update_AAort_(self, bd):
        """ Update projection of states to be subtracted from psi. """
        Aort = []
        nl, nr = bd
        inds = ((-0, 1), (1, -1, -2,  2), (2, -3)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        for ii in range(len(self.ort)):
            AA = self.ort[ii].merge_two_sites(bd)
            Aort.append(tensor.ncon([self.Fort[ii][(nl - 1, nl)], AA, self.Fort[ii][(nr + 1, nr)]], inds))
        self._temp['Aort'] = Aort

    def _project_ort(self, A):
        for ii in range(len(self.ort)):
            x = tensor.vdot(self._temp['Aort'][ii], A)
            A = A.apxb(self._temp['Aort'][ii], -x)
        return A


class Env2(_EnvParent):
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, ket=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra: mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        ket: mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket)
        self.nr_layers=2

        # left boundary
        config = self.bra[0].config
        self.F= [{}]
        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj()]
        self.F[-1][(-1, 0)] = initialize.ones(config=config, legs=legs)
        # right boundary
        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[-1][(self.N, self.N - 1)] = initialize.ones(config=config, legs=legs)


    def factor(self):
        return self.bra.factor * self.ket.factor

    def Heff1(self, x, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A: tensor
            site tensor

        n: int
            index of corresponding site

        Returns
        -------
        out: tensor
            Heff1 * A
        """
        inds = ((-0, 1), (1, -1, 2), (2, -2)) if self.nr_phys == 1 else ((-0, 1), (1, -1, 2, -3), (2, -2))
        return tensor.ncon([self.F[-1][(n - 1, n)], x, self.F[-1][(n + 1, n)]], inds)

    def Heff2(self, AA, bd):
        """ Heff2 @ AA """
        n1, n2 = bd
        axes = (0, (1, 2), 3) if AA.ndim == 4 else (0, (1, 2, 3, 5), 4)
        temp = AA.fuse_legs(axes=axes)
        temp = self.F[-1][(n1 - 1, n1)] @ temp @ self.F[-1][(n2 + 1, n2)]
        temp = temp.unfuse_legs(axes=1)
        if temp.ndim == 6:
            temp = temp.transpose(axes=(0, 1, 2, 3, 5, 4))
        return temp

    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at bd bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.

        Returns
        -------
        overlap: float or complex
        """
        if bd is None:
            bd = (-1, 0)
        axes = ((0, 1), (1, 0))
        return self.factor() * self.F[-1][bd].tensordot(self.F[-1][bd[::-1]], axes=axes).to_number()

    def update_env_(self, n, to='last'):
        _update2_(n, self.F, self.bra, self.ket, to, self.nr_phys)
        # for ii in range(len(self.ort)):
        _update2_(n, self.Fort, self.bra, self.ort, to, self.nr_phys)

    def update_env_op_(self, n, op, to='first'):
        """ Contractions for 2-layer environment update. """
        if to == 'first':
            temp = tensor.tensordot(self.ket[n], self.F[-1][(n + 1, n)], axes=(2, 0))
            op = op.add_leg(axis=0, s=1)
            temp = tensor.tensordot(op, temp, axes=(2, 1))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            axes = ((0, 2), (1, 2)) if self.nr_phys == 1 else ((0, 3, 2), (1, 2, 3))
            self.F[-1][(n, n - 1)] = tensor.tensordot(temp, self.bra[n].conj(), axes=axes)
        else:  # to == 'last'
            op = op.add_leg(axis=0, s=1)
            temp = tensor.tensordot(op, self.ket[n], axes=((2, 1)))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            temp = tensor.tensordot(self.F[-1][(n - 1, n)], temp, axes=((1, 1)))
            axes = ((0, 1), (0, 1)) if self.nr_phys == 1 else ((0, 1, 3), (0, 1, 3))
            self.F[-1][(n, n + 1)] = tensor.tensordot(self.bra[n].conj(), temp, axes=axes)


class Env3(_EnvParent):
    """
    The class combines environments of mps+mpo+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, op: Optional[MpsMpo | Sequence[tuple(MpsMpo,number)]] = None , ket=None, on_aux=False, project=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm op} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra: mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        op: mps
            mps for operator op.
        ket: mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket, project)
        self.op = [(op,1.)] if type(op)==MpsMpo else op
        self.nr_layers = 3
        self.on_aux = on_aux
        if any([_op.N != self.N for _op,_ in self.op]):
            raise YastnError('all MPO operators and state should have the same number of sites.')

        config = self.ket[0].config
        self.F= [{} for _ in self.op]
        for i,op in enumerate([_op for _op,_ in self.op]):
             # left boundary
            l_legs = [self.bra.virtual_leg('first'), op.virtual_leg('first').conj(), self.ket.virtual_leg('first').conj()]
            self.F[i][(-1, 0)] = initialize.ones(config=config, legs=l_legs)
            # right boundary
            r_legs = [self.ket.virtual_leg('last').conj(), op.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
            self.F[i][(self.N, self.N - 1)] = initialize.ones(config=config, legs=r_legs)


    def factor(self)->number:
        return self.bra.factor * self.ket.factor
        # return [self.bra.factor * _op.factor * self.ket.factor for _op,_ in self.op]

    def Heff0(self, C, bd):
        r"""
        Action of Heff on central block.

        Parameters
        ----------
        C: tensor
            a central block
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out: tensor
            Heff0 @ C
        """
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        _heff0= lambda _op,_amp,i=0: tensor.ncon([self.F[i][bd], _amp*_op.factor*C, self.F[i][ibd]], ((-0, 2, 1), (1, 3), (3, 2, -1)))
        return sum([_heff0(*op,i) for i,op in enumerate(self.op[1:],start=1)], _heff0(*self.op[0]))


    def Heff1(self, A, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A: tensor
            site tensor

        n: int
            index of corresponding site

        Returns
        -------
        out: tensor
            Heff1 @ A
        """
        nl, nr = n - 1, n + 1
        def heff1(op,amp,F):
            tmp = self._project_ort(A)
            if self.nr_phys == 1:
                tmp = tmp @ F[(nr, n)]
                tmp = op[n]._attach_23(tmp)
                tmp = tensor.ncon([F[(nl, n)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
            elif self.nr_phys == 2 and not self.on_aux:
                tmp = tmp.fuse_legs(axes=((0, 3), 1, 2))
                tmp = tmp @ F[(nr, n)]
                tmp = op[n]._attach_23(tmp)
                tmp = tmp.unfuse_legs(axes=0)
                tmp = tensor.ncon([F[(nl, n)], tmp], ((-0, 1, 2), (2, -3, 1, -2, -1)))
            else:  # if self.nr_phys == 2 and self.on_aux:
                tmp = tmp.fuse_legs(axes=(0, (1, 2), 3))
                tmp = tensor.ncon([tmp, F[(nl, n)]], ((1, -0, -1), (-3, -2, 1)))
                tmp = op[n]._attach_01(tmp)
                tmp = tmp.unfuse_legs(axes=0)
                tmp = tensor.ncon([tmp, F[(nr, n)]], ((-1, 1, -0, 2, -3), (1, 2, -2)))
            return amp * op.factor * self._project_ort(tmp)
        return sum([heff1(*op,self.F[i]) for i,op in enumerate(self.op[1:],start=1)],heff1(*self.op[0],self.F[0]))


    def Heff2(self, AA, bd):
        r"""
        Action of Heff on central block.

        Parameters
        ----------
        AA: tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectivly into 1-site update.
        bd: tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out: tensor
            Heff2 * AA
        """
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1

        def heff2(op,amp,F):
            tmp = self._project_ort(AA)
            if self.nr_phys == 1:
                tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
                tmp = tmp @ F[(nr, n2)]
                tmp = op[n2]._attach_23(tmp)
                tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
                tmp = tmp.unfuse_legs(axes=0)
                tmp = op[n1]._attach_23(tmp)
                tmp = tensor.ncon([F[(nl, n1)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
                tmp = tmp.unfuse_legs(axes=2)
            elif self.nr_phys == 2 and not self.on_aux:
                tmp = tmp.fuse_legs(axes=((0, 2, 5), 1, 3, 4))
                tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
                tmp = tmp @ F[(nr, n2)]
                tmp = op[n2]._attach_23(tmp)
                tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
                tmp = tmp.unfuse_legs(axes=0)
                tmp = op[n1]._attach_23(tmp)
                tmp = tmp.unfuse_legs(axes=0)
                tmp = tensor.ncon([F[(nl, n1)], tmp], ((-0, 1, 2), (2, -2, -4, 1, -3, -1)))
                tmp = tmp.unfuse_legs(axes=3)
            else:  # if self.nr_phys == 2 and self.on_aux:
                tmp = tmp.fuse_legs(axes=(0, 2, (1, 3, 4), 5))
                tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
                tmp = tensor.ncon([tmp, F[(nl, n1)]], ((1, -1, -0), (-3, -2, 1)))
                tmp = op[n1]._attach_01(tmp)
                tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
                tmp = tmp.unfuse_legs(axes=0)
                tmp = op[n2]._attach_01(tmp)
                tmp = tmp.unfuse_legs(axes=0)
                tmp = tensor.ncon([tmp, F[(nr, n2)]], ((-1, -2, 1, 2, -0, -4), (1, 2, -3)))
                tmp = tmp.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3, 4, 5))
            return amp * op.factor * self._project_ort(tmp)
        return sum([heff2(*op,self.F[i]) for i,op in enumerate(self.op[1:],start=1)],heff2(*self.op[0],self.F[0]))


    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at bd bond.

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.

        Returns
        -------
        overlap: float or complex
        """
        if bd is None:
            bd = (-1, 0)
        axes = ((0, 1, 2), (2, 1, 0))
        _measure= lambda i,op,amp: amp * op.factor * self.F[i][bd].tensordot(self.F[i][bd[::-1]], axes=axes).to_number()
        return self.factor() * sum([_measure(i,*op) for i,op in enumerate(self.op[1:],start=1)], _measure(0,*self.op[0]))


    def update_env_(self, n, to='last'):
        _update3_(n, self.F, self.bra, self.op, self.ket, to, self.nr_phys, self.on_aux)
        # for ii in range(len(self.ort)):
        _update2_(n, self.Fort, self.bra, self.ort, to, self.nr_phys)


    def update_A(self, n, du, opts, normalize=True):
        """ Updates env.ket[n] by exp(du Heff1). """
        if n in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][n]
        f = lambda x: self.Heff1(x, n)
        self.ket[n], info = expmv(f, self.ket[n], du, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][n] = info['ncv']

    def update_C(self, du, opts, normalize=True):
        """ Updates env.ket[bd] by exp(du Heff0). """
        bd = self.ket.pC
        if bd[0] != -1 and bd[1] != self.N:  # do not update central block outsite of the chain
            if bd in self._temp['expmv_ncv']:
                opts['ncv'] = self._temp['expmv_ncv'][bd]
            f = lambda x: self.Heff0(x, bd)
            self.ket.A[bd], info = expmv(f, self.ket[bd], du, **opts, normalize=normalize, return_info=True)
            self._temp['expmv_ncv'][bd] = info['ncv']

    def update_AA(self, bd, du, opts, opts_svd, normalize=True):
        """ Merge two sites given in bd into AA, updates AA by exp(du Heff2) and unmerge the sites. """
        ibd = bd[::-1]
        if ibd in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][ibd]
        AA = self.ket.merge_two_sites(bd)
        f = lambda v: self.Heff2(v, bd)
        AA, info = expmv(f, AA, du, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][ibd] = info['ncv']
        self.ket.unmerge_two_sites_(AA, bd, opts_svd)

    def enlarge_bond(self, bd, opts_svd):
        if bd[0] < 0 or bd[1] >= self.N:  # do not enlarge bond outside of the chain
            return False
        AL = self.ket[bd[0]]
        AR = self.ket[bd[1]]
        if any(op[bd[0]].get_legs(axes=1).t != AL.get_legs(axes=1).t for op,amp in self.op) or \
           any(op[bd[1]].get_legs(axes=1).t != AR.get_legs(axes=1).t for op,amp in self.op):
            return True  # true if some charges are missing on physical legs of psi

        AL = AL.fuse_legs(axes=((0, 1), 2))
        AR = AR.fuse_legs(axes=(0, (1, 2)))
        shapeL = AL.get_shape()
        shapeR = AR.get_shape()
        if shapeL[0] == shapeL[1] or shapeR[0] == shapeR[1] or \
           ('D_total' in opts_svd and shapeL[1] >= opts_svd['D_total']):
            return False  # maximal bond dimension
        if 'tol' in opts_svd:
            _, R0 = tensor.qr(AL, axes=(0, 1), sQ=1)
            _, R1 = tensor.qr(AR, axes=(1, 0), Raxis=1, sQ=-1)
            S = tensor.svd(R0 @ R1, compute_uv=False)
            if any(S[t][-1] > opts_svd['tol'] * 1.1 for t in S.struct.t):
                return True
        return False


def _update2_(n, F : Sequence[Dict[tuple(int),Tensor]], bra : MpsMpo, kets : MpsMpo | Sequence[MpsMpo], to, nr_phys):
    """
    Contractions for 2-layer environment update. 

    Updates sets of *n* environments ``F`` associated with *n* operators ``op``
    
    * if bra and kets are both MpsMpo, F is assumed to be length 1
    """
    if type(kets)==MpsMpo:
        assert len(F)==1
        kets=[kets]
    elif type(kets)!=MpsMpo:
        assert len(F)==len(kets)
    
    for _f,ket in zip(F,kets):
        if to == 'first':
            inds = ((-0, 2, 1), (1, 3), (-1, 2, 3)) if nr_phys == 1 else ((-0, 2, 1, 4), (1, 3), (-1, 2, 3, 4))
            _f[(n, n - 1)] = tensor.ncon([ket[n], _f[(n + 1, n)], bra[n].conj()], inds)
        elif to == 'last':
            inds = ((2, 3, -0), (2, 1), (1, 3, -1)) if nr_phys == 1 else ((2, 3, -0, 4), (2, 1), (1, 3, -1, 4))
            _f[(n, n + 1)] = tensor.ncon([bra[n].conj(), _f[(n - 1, n)], ket[n]], inds)


def _update3_(n, F : Sequence[Dict[tuple(int),Tensor]], bra, op : Sequence[tuple(MpsMpo,number)], ket, to, nr_phys, on_aux):
    r"""
    Contractions for 3-layer environment update.

    Updates sets of *n* environments ``F`` associated with *n* operators ``op``
    """
    for _op, _f in zip([mpo for mpo,amp in op],F):
        if nr_phys == 1 and to == 'last':
            tmp = tensor.ncon([bra[n].conj(), _f[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
            tmp = _op[n]._attach_01(tmp)
            _f[(n, n + 1)] = tensor.ncon([tmp, ket[n]], ((-0, -1, 1, 2), (1, 2, -2)))
        elif nr_phys == 1 and to == 'first':
            tmp = ket[n] @ _f[(n + 1, n)]
            tmp = _op[n]._attach_23(tmp)
            _f[(n, n - 1)] = tensor.ncon([tmp, bra[n].conj()], ((-0, -1, 1, 2), (-2, 2, 1)))
        elif nr_phys == 2 and not on_aux and to == 'last':
            bA = bra[n].fuse_legs(axes=(0, 1, (2, 3)))
            tmp = tensor.ncon([bA.conj(), _f[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
            tmp = _op[n]._attach_01(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            _f[(n, n + 1)] = tensor.ncon([tmp, ket[n]], ((-0, 3, -1, 1, 2), (1, 2, -2, 3)))
        elif nr_phys == 2 and not on_aux and to == 'first':
            kA = ket[n].fuse_legs(axes=((0, 3), 1, 2))
            tmp = tensor.ncon([kA, _f[(n + 1, n)]], ((-0, -1, 1), (1, -2, -3)))
            tmp = _op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            _f[(n, n - 1)] = tensor.ncon([tmp, bra[n].conj()], ((-0, 3, -1, 1, 2), (-2, 2, 1, 3)))
        elif nr_phys == 2 and on_aux and to == 'last':
            tmp = tensor.ncon([ket[n], _f[(n - 1, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
            tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
            tmp = _op[n]._attach_01(tmp)
            bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
            _f[(n, n + 1)] = tensor.ncon([bA.conj(), tmp], ((1, -0, 2), (-2, -1, 1, 2)))
        else: # nr_phys == 2 and on_aux and to == 'first':
            bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
            tmp = tensor.ncon([bA.conj(), _f[(n + 1, n)]], ((-0, 1, -1), (-3, -2, 1)))
            tmp = _op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            _f[(n, n - 1)] = tensor.ncon([ket[n], tmp], ((-0, 1, 2, 3), (-2, 1, -1, 2, 3)))


