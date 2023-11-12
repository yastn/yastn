""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from __future__ import annotations
from ... import tensor, initialize, YastnError, expmv
from itertools import groupby


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


def measure_mpo(bra, op, ket) -> number:
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`.

    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length,
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra: yastn.tn.mps.MpsMpo
        An MPS which will be conjugated.

    op: yastn.tn.mps.MpsMpo
        Operator written as MPO.

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


class _EnvParent:
    def __init__(self, bra=None, ket=None, project=None) -> None:
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.N = ket.N
        self.nr_phys = ket.nr_phys
        self.nr_layers = 2
        self.F = {}  # dict for environments
        self.ort = [] if project is None else project
        self.Fort = [{} for _ in range(len(self.ort))]
        self._temp = {}
        self.reset_temp_()

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YastnError('MpsMpo for bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YastnError('MpsMpo for bra and ket should have the same number of sites.')

        config = self.ket[0].config
        for ii in range(len(self.ort)):
            legs = [self.ort[ii].virtual_leg('first'), self.ket.virtual_leg('first').conj()]
            self.Fort[ii][(-1, 0)] = initialize.ones(config=config, legs=legs)
            legs = [self.ket.virtual_leg('last').conj(), self.ort[ii].virtual_leg('last')]
            self.Fort[ii][(self.N, self.N - 1)] = initialize.ones(config=config, legs=legs)

    def reset_temp_(self):
        """ Reset temporary objects stored to speed-up some simulations. """
        self._temp = {'Aort': [], 'op_2site': {}, 'expmv_ncv': {}}

    def setup_(self, to='last'):
        r"""
        Setup all environments in the direction given by to.

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
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

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
        axes = ((0, 1), (1, 0)) if self.nr_layers == 2 else ((0, 1, 2), (2, 1, 0))
        return self.factor() * self.F[bd].tensordot(self.F[bd[::-1]], axes=axes).to_number()

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
        if self.nr_layers == 2:
            _update2_(n, self.F, self.bra, self.ket, to, self.nr_phys)
        else:
            _update3_(n, self.F, self.bra, self.op, self.ket, to, self.nr_phys, self.on_aux)
        for ii in range(len(self.ort)):
            _update2_(n, self.Fort[ii], self.bra, self.ort[ii], to, self.nr_phys)

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

        # left boundary
        config = self.bra[0].config
        legs = [self.bra.virtual_leg('first'), self.ket.virtual_leg('first').conj()]
        self.F[(-1, 0)] = initialize.ones(config=config, legs=legs)
        # right boundary
        legs = [self.ket.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[(self.N, self.N - 1)] = initialize.ones(config=config, legs=legs)

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
        return tensor.ncon([self.F[(n - 1, n)], x, self.F[(n + 1, n)]], inds)

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
        """ Contractions for 2-layer environment update. """
        if to == 'first':
            temp = tensor.tensordot(self.ket[n], self.F[(n + 1, n)], axes=(2, 0))
            op = op.add_leg(axis=0, s=1)
            temp = tensor.tensordot(op, temp, axes=(2, 1))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            axes = ((0, 2), (1, 2)) if self.nr_phys == 1 else ((0, 3, 2), (1, 2, 3))
            self.F[(n, n - 1)] = tensor.tensordot(temp, self.bra[n].conj(), axes=axes)
        else:  # to == 'last'
            op = op.add_leg(axis=0, s=1)
            temp = tensor.tensordot(op, self.ket[n], axes=((2, 1)))
            temp = temp.swap_gate(axes=(0, 2))
            temp = temp.remove_leg(axis=0)
            temp = tensor.tensordot(self.F[(n - 1, n)], temp, axes=((1, 1)))
            axes = ((0, 1), (0, 1)) if self.nr_phys == 1 else ((0, 1, 3), (0, 1, 3))
            self.F[(n, n + 1)] = tensor.tensordot(self.bra[n].conj(), temp, axes=axes)


class Env3(_EnvParent):
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, op=None, ket=None, on_aux=False, project=None):
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
        self.op = op
        self.nr_layers = 3
        self.on_aux = on_aux
        if self.op.N != self.N:
            raise YastnError('MPO operator and state should have the same number of sites.')

        # left boundary
        config = self.ket[0].config
        legs = [self.bra.virtual_leg('first'), self.op.virtual_leg('first').conj(), self.ket.virtual_leg('first').conj()]
        self.F[(-1, 0)] = initialize.ones(config=config, legs=legs)

        # right boundary
        legs = [self.ket.virtual_leg('last').conj(), self.op.virtual_leg('last').conj(), self.bra.virtual_leg('last')]
        self.F[(self.N, self.N - 1)] = initialize.ones(config=config, legs=legs)

    def factor(self):
        return self.bra.factor * self.op.factor * self.ket.factor

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
        C = self.op.factor * C
        return tensor.ncon([self.F[bd], C, self.F[ibd]], ((-0, 2, 1), (1, 3), (3, 2, -1)))


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
        tmp = self._project_ort(A)
        if self.nr_phys == 1:
            tmp = tmp @ self.F[(nr, n)]
            tmp = self.op[n]._attach_23(tmp)
            tmp = tensor.ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
        elif self.nr_phys == 2 and not self.on_aux:
            tmp = tmp.fuse_legs(axes=((0, 3), 1, 2))
            tmp = tmp @ self.F[(nr, n)]
            tmp = self.op[n]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            tmp = tensor.ncon([self.F[(nl, n)], tmp], ((-0, 1, 2), (2, -3, 1, -2, -1)))
        else:  # if self.nr_phys == 2 and self.on_aux:
            tmp = tmp.fuse_legs(axes=(0, (1, 2), 3))
            tmp = tensor.ncon([tmp, self.F[(nl, n)]], ((1, -0, -1), (-3, -2, 1)))
            tmp = self.op[n]._attach_01(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            tmp = tensor.ncon([tmp, self.F[(nr, n)]], ((-1, 1, -0, 2, -3), (1, 2, -2)))
        return self.op.factor * self._project_ort(tmp)


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

        tmp = self._project_ort(AA)
        if self.nr_phys == 1:
            tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
            tmp = tmp @ self.F[(nr, n2)]
            tmp = self.op[n2]._attach_23(tmp)
            tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
            tmp = tmp.unfuse_legs(axes=0)
            tmp = self.op[n1]._attach_23(tmp)
            tmp = tensor.ncon([self.F[(nl, n1)], tmp], ((-0, 1, 2), (2, 1, -2, -1)))
            tmp = tmp.unfuse_legs(axes=2)
        elif self.nr_phys == 2 and not self.on_aux:
            tmp = tmp.fuse_legs(axes=((0, 2, 5), 1, 3, 4))
            tmp = tmp.fuse_legs(axes=((0, 1), 2, 3))
            tmp = tmp @ self.F[(nr, n2)]
            tmp = self.op[n2]._attach_23(tmp)
            tmp = tmp.fuse_legs(axes=(0, 1, (3, 2)))
            tmp = tmp.unfuse_legs(axes=0)
            tmp = self.op[n1]._attach_23(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            tmp = tensor.ncon([self.F[(nl, n1)], tmp], ((-0, 1, 2), (2, -2, -4, 1, -3, -1)))
            tmp = tmp.unfuse_legs(axes=3)
        else:  # if self.nr_phys == 2 and self.on_aux:
            tmp = tmp.fuse_legs(axes=(0, 2, (1, 3, 4), 5))
            tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
            tmp = tensor.ncon([tmp, self.F[(nl, n1)]], ((1, -1, -0), (-3, -2, 1)))
            tmp = self.op[n1]._attach_01(tmp)
            tmp = tmp.fuse_legs(axes=(0, 1, (2, 3)))
            tmp = tmp.unfuse_legs(axes=0)
            tmp = self.op[n2]._attach_01(tmp)
            tmp = tmp.unfuse_legs(axes=0)
            tmp = tensor.ncon([tmp, self.F[(nr, n2)]], ((-1, -2, 1, 2, -0, -4), (1, 2, -3)))
            tmp = tmp.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3, 4, 5))
        return self.op.factor * self._project_ort(tmp)

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
        if self.op[bd[0]].get_legs(axes=1).t != AL.get_legs(axes=1).t or \
           self.op[bd[1]].get_legs(axes=1).t != AR.get_legs(axes=1).t:
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


def _update2_(n, F, bra, ket, to, nr_phys):
    """ Contractions for 2-layer environment update. """
    if to == 'first':
        inds = ((-0, 2, 1), (1, 3), (-1, 2, 3)) if nr_phys == 1 else ((-0, 2, 1, 4), (1, 3), (-1, 2, 3, 4))
        F[(n, n - 1)] = tensor.ncon([ket[n], F[(n + 1, n)], bra[n].conj()], inds)
    elif to == 'last':
        inds = ((2, 3, -0), (2, 1), (1, 3, -1)) if nr_phys == 1 else ((2, 3, -0, 4), (2, 1), (1, 3, -1, 4))
        F[(n, n + 1)] = tensor.ncon([bra[n].conj(), F[(n - 1, n)], ket[n]], inds)


def _update3_(n, F, bra, op, ket, to, nr_phys, on_aux):
    if nr_phys == 1 and to == 'last':
        tmp = tensor.ncon([bra[n].conj(), F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
        tmp = op[n]._attach_01(tmp)
        F[(n, n + 1)] = tensor.ncon([tmp, ket[n]], ((-0, -1, 1, 2), (1, 2, -2)))
    elif nr_phys == 1 and to == 'first':
        tmp = ket[n] @ F[(n + 1, n)]
        tmp = op[n]._attach_23(tmp)
        F[(n, n - 1)] = tensor.ncon([tmp, bra[n].conj()], ((-0, -1, 1, 2), (-2, 2, 1)))
    elif nr_phys == 2 and not on_aux and to == 'last':
        bA = bra[n].fuse_legs(axes=(0, 1, (2, 3)))
        tmp = tensor.ncon([bA.conj(), F[(n - 1, n)]], ((1, -1, -0), (1, -2, -3)))
        tmp = op[n]._attach_01(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n + 1)] = tensor.ncon([tmp, ket[n]], ((-0, 3, -1, 1, 2), (1, 2, -2, 3)))
    elif nr_phys == 2 and not on_aux and to == 'first':
        kA = ket[n].fuse_legs(axes=((0, 3), 1, 2))
        tmp = tensor.ncon([kA, F[(n + 1, n)]], ((-0, -1, 1), (1, -2, -3)))
        tmp = op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n - 1)] = tensor.ncon([tmp, bra[n].conj()], ((-0, 3, -1, 1, 2), (-2, 2, 1, 3)))
    elif nr_phys == 2 and on_aux and to == 'last':
        tmp = tensor.ncon([ket[n], F[(n - 1, n)]], ((1, -4, -0, -1), (-3, -2, 1)))
        tmp = tmp.fuse_legs(axes=(0, 1, 2, (3, 4)))
        tmp = op[n]._attach_01(tmp)
        bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        F[(n, n + 1)] = tensor.ncon([bA.conj(), tmp], ((1, -0, 2), (-2, -1, 1, 2)))
    else: # nr_phys == 2 and on_aux and to == 'first':
        bA = bra[n].fuse_legs(axes=((0, 1), 2, 3))
        tmp = tensor.ncon([bA.conj(), F[(n + 1, n)]], ((-0, 1, -1), (-3, -2, 1)))
        tmp = op[n]._attach_23(tmp)
        tmp = tmp.unfuse_legs(axes=0)
        F[(n, n - 1)] = tensor.ncon([ket[n], tmp], ((-0, 1, 2, 3), (-2, 1, -1, 2, 3)))
