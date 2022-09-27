""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from yast import ncon, tensordot, expmv, vdot, qr, svd, ones
from ._mps import YampsError


def measure_overlap(bra, ket):
    r"""
    Calculate overlap :math:`\langle \textrm{bra}|\textrm{ket} \rangle`. 
    Conjugate of MPS :code:`bra` is computed internally.
    
    MPSs :code:`bra` and :code:`ket` must have matching length, 
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra : yamps.MpsMpo
        An MPS which will be conjugated.

    ket : yamps.MpsMpo

    Returns
    -------
    scalar
    """
    env = Env2(bra=bra, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op, ket):
    r"""
    Calculate expectation value :math:`\langle \textrm{bra}|\textrm{op}|\textrm{ket} \rangle`. 
    Conjugate of MPS :code:`bra` is computed internally.
    MPSs :code:`bra`, :code:`ket`, and MPO :code:`op` must have matching length, 
    physical dimensions, and symmetry.

    Parameters
    -----------
    bra : yamps.MpsMpo
        An MPS which will be conjugated.

    op : yamps.MpsMpo
        Operator written as MPO.

    ket : yamps.MpsMpo

    Returns
    -------
    scalar
    """
    env = Env3(bra=bra, op=op, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


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
        self.reset_temp()

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YampsError('bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YampsError('bra and ket should have the same number of sites.')

        config = self.ket.A[0].config
        for ii in range(len(self.ort)):
            legs = [self.ort[ii].get_leftmost_leg(), self.ket.get_leftmost_leg().conj()]
            self.Fort[ii][(-1, 0)] = ones(config=config, legs=legs)
            legs = [self.ket.get_rightmost_leg().conj(), self.ort[ii].get_rightmost_leg()]
            self.Fort[ii][(self.N, self.N - 1)] = ones(config=config, legs=legs)

    def reset_temp(self):
        """ Reset temporary objects stored to speed-up some simulations. """
        self._temp = {'Aort': [], 'op_2site': {}, 'expmv_ncv': {}}

    def setup(self, to='last'):
        r"""
        Setup all environments in the direction given by to.

        Parameters
        ----------
        to : str
            'first' or 'last'.
        """
        for n in self.ket.sweep(to=to):
            self.update_env(n, to=to)
        return self

    def clear_site(self, *args):
        r""" Clear environments pointing from sites which indices are provided in args """
        for n in args:
            self.F.pop((n, n - 1), None)
            self.F.pop((n, n + 1), None)

    def measure(self, bd=None):
        r"""
        Calculate overlap between environments at bd bond

        Parameters
        ----------
        bd: tuple
            index of bond at which to calculate overlap.

        Returns
        -------
        overlap : float or complex
        """
        if bd is None:
            bd = (-1, 0)
        axes = ((0, 1), (1, 0)) if self.nr_layers == 2 else ((0, 1, 2), (2, 1, 0))
        return tensordot(self.F[bd], self.F[bd[::-1]], axes=axes).to_number()

    def project_ket_on_bra(self, n):
        r"""Project ket on a n-th site of bra.

        It is equal to the overlap <bra|op|ket> up to the contribution from n-th site of bra.

        Parameters
        ----------
        n : int
            index of site

        Returns
        -------
        out : tensor
            result of projection
        """
        return self.Heff1(self.ket.A[n], n)

    def update_env(self, n, to='last'):
        r"""
        Update environment including site n, in the direction given by to.

        Parameters
        ----------
        n: int
            index of site to include to the environment

        to : str
            'first' or 'last'.
        """
        if self.nr_layers == 2:
            _update2(n, self.F, self.bra, self.ket, to, self.nr_phys)
        else:
            _update3(n, self.F, self.bra, self.op, self.ket, to, self.nr_phys, self.on_aux)
        for ii in range(len(self.ort)):
            _update2(n, self.Fort[ii], self.bra, self.ort[ii], to, self.nr_phys)

    def update_Aort(self, n):
        """ Update projection of states to be project to on psi. """
        Aort = []
        for ii in range(len(self.ort)):
            T1 = tensordot(self.Fort[ii][(n - 1, n)], self.ort[ii].A[n], axes=(1, 0))
            Aort.append(tensordot(T1, self.Fort[ii][(n + 1, n)], axes=(self.nr_phys + 1, 0)))
        self._temp['Aort'] = Aort

    def update_AAort(self, bd):
        """ Update projection of states to be project to on psi. """
        Aort = []
        nl, nr = bd
        for ii in range(len(self.ort)):
            AA = self.ort[ii].merge_two_sites(bd)
            T1 = tensordot(self.Fort[ii][(nl - 1, nl)], AA, axes=(1, 0))
            Aort.append(tensordot(T1, self.Fort[ii][(nr + 1, nr)], axes=(self.nr_phys + 1, 0)))
        self._temp['Aort'] = Aort

    def _project_ort(self, A):
        for ii in range(len(self.ort)):
            x = vdot(self._temp['Aort'][ii], A)
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
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket)

        # left boundary
        config = self.bra.A[0].config
        legs = [self.bra.get_leftmost_leg(), self.ket.get_leftmost_leg().conj()]
        self.F[(-1, 0)] = ones(config=config, legs=legs)
        # right boundary
        legs = [self.ket.get_rightmost_leg().conj(), self.bra.get_rightmost_leg()]
        self.F[(self.N, self.N - 1)] = ones(config=config, legs=legs)

    def Heff1(self, x, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A : tensor
            site tensor

        n : int
            index of corresponding site

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        return tensordot(tensordot(self.F[(n - 1, n)], x, axes=(1, 0)), self.F[(n + 1, n)], axes=(self.nr_phys + 1, 0))


class Env3(_EnvParent):
    """
    The class combines environments of mps+mps for calculation of expectation values, overlaps, etc.
    """

    def __init__(self, bra=None, op=None, ket=None, on_aux=False, project=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm opp} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        opp : mps
            mps for operator opp.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket, project)
        self.op = op
        self.nr_layers = 3
        self.on_aux = on_aux
        if self.op.N != self.N:
            raise YampsError('op should should have the same number of sites as ket.')

        # left boundary
        config = self.ket.A[0].config
        legs = [self.bra.get_leftmost_leg(), self.op.get_leftmost_leg().conj(), self.ket.get_leftmost_leg().conj()]
        self.F[(-1, 0)] = ones(config=config, legs=legs)

        # right boundary
        legs = [self.ket.get_rightmost_leg().conj(), self.op.get_rightmost_leg().conj(), self.bra.get_rightmost_leg()]
        self.F[(self.N, self.N - 1)] = ones(config=config, legs=legs)

    def Heff0(self, C, bd):
        r"""
        Action of Heff on central site.

        Parameters
        ----------
        C : tensor
            a central site
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff0 * C
        """
        bd, ibd = (bd[::-1], bd) if bd[1] < bd[0] else (bd, bd[::-1])
        return tensordot(tensordot(self.F[bd], C, axes=(2, 0)), self.F[ibd], axes=((1, 2), (1, 0)))

    def Heff1(self, A, n):
        r"""
        Action of Heff on a single site mps tensor.

        Parameters
        ----------
        A : tensor
            site tensor

        n : int
            index of corresponding site

        Returns
        -------
        out : tensor
            Heff1 * A
        """
        nl, nr = n - 1, n + 1
        A = self._project_ort(A)
        if self.nr_phys == 1:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
        elif self.nr_phys == 2 and not self.on_aux:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        else:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        T1 = self._project_ort(T1)
        return T1

    def Heff2(self, AA, bd):
        r"""Action of Heff on central site.

        Parameters
        ----------
        AA : tensor
            merged tensor for 2 sites.
            Physical legs should be fused turning it effectivly into 1-site update.
        bd : tuple
            index of bond on which it acts, e.g. (1, 2) [or (2, 1) -- it is ordered]

        Returns
        -------
        out : tensor
            Heff2 * AA
        """
        n1, n2 = bd if bd[0] < bd[1] else bd[::-1]
        bd, nl, nr = (n1, n2), n1 - 1, n2 + 1

        if bd not in self._temp['op_2site']:
            OO = tensordot(self.op.A[n1], self.op.A[n2], axes=(3, 0))
            self._temp['op_2site'][bd] = OO.fuse_legs(axes=(0, (1, 3), (2, 4), 5))
        OO = self._temp['op_2site'][bd]

        AA = self._project_ort(AA)
        if self.nr_phys == 1:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
        if self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))

    def update_A(self, n, dt, opts, normalize=True):
        """ Updates env.ket.A[n] by exp(dt Heff1). """
        if n in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][n]
        f = lambda x: self.Heff1(x, n)
        self.ket.A[n], info = expmv(f, self.ket.A[n], dt, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][n] = info['ncv']

    def update_C(self, dt, opts, normalize=True):
        """ Updates env.ket.A[bd] by exp(dt Heff0). """
        bd = self.ket.pC
        if bd[0] != -1 and bd[1] != self.N:  # do not update central sites outsite of the chain
            if bd in self._temp['expmv_ncv']:
                opts['ncv'] = self._temp['expmv_ncv'][bd]
            f = lambda x: self.Heff0(x, bd)
            self.ket.A[bd], info = expmv(f, self.ket.A[bd], dt, **opts, normalize=normalize, return_info=True)
            self._temp['expmv_ncv'][bd] = info['ncv']

    def update_AA(self, bd, dt, opts, opts_svd, normalize=True):
        """ Merge two sites given in bd into AA, updates AA by exp(dt Heff2) and unmerge the sites. """
        ibd = bd[::-1]
        if ibd in self._temp['expmv_ncv']:
            opts['ncv'] = self._temp['expmv_ncv'][ibd]
        AA = self.ket.merge_two_sites(bd)
        f = lambda v: self.Heff2(v, bd)
        AA, info = expmv(f, AA, dt, **opts, normalize=normalize, return_info=True)
        self._temp['expmv_ncv'][ibd] = info['ncv']
        self.ket.unmerge_two_sites(AA, bd, opts_svd)

    def enlarge_bond(self, bd, opts_svd):
        if bd[0] < 0 or bd[1] >= self.N:  # do not enlarge bond outside of the chain
            return False
        AL = self.ket.A[bd[0]]
        AR = self.ket.A[bd[1]]
        if self.op.A[bd[0]].get_legs(axis=1).t != AL.get_legs(axis=1).t or \
           self.op.A[bd[1]].get_legs(axis=1).t != AR.get_legs(axis=1).t:
            return True  # true if some charges are missing on physical legs of psi

        AL = AL.fuse_legs(axes=((0, 1), 2))
        AR = AR.fuse_legs(axes=(0, (1, 2)))
        shapeL = AL.get_shape()
        shapeR = AR.get_shape()
        if shapeL[0] == shapeL[1] or shapeR[0] == shapeR[1] or \
           ('D_total' in opts_svd and shapeL[0] >= opts_svd['D_total']):
            return False  # maximal bond dimension
        if 'tol' in opts_svd:
            _, R0 = qr(AL, axes=(0, 1), sQ=-1)
            _, R1 = qr(AR, axes=(1, 0), Raxis=1, sQ=1)
            _, S, _ = svd(R0 @ R1)
            if any(S[t][-1] > opts_svd['tol'] * 1.1 for t in S.struct.t):
                return True
        return False


def _update2(n, F, bra, ket, to, nr_phys):
    """ Contractions for 2-layer environment update. """
    if to == 'first':
        if nr_phys == 1:
            T1 = tensordot(ket.A[n], F[(n + 1, n)], axes=(2, 0))
            F[(n, n - 1)] = tensordot(T1, bra.A[n], axes=((1, 2), (1, 2)), conj=(0, 1))
        else:
            T1 = tensordot(ket.A[n], F[(n + 1, n)], axes=(3, 0))
            F[(n, n - 1)] = tensordot(T1, bra.A[n], axes=((1, 2, 3), (1, 2, 3)), conj=(0, 1))
    elif to == 'last':
        if nr_phys == 1:
            T1 = tensordot(F[(n - 1, n)], bra.A[n], axes=(0, 0), conj=(0, 1))
            F[(n, n + 1)] = tensordot(T1, ket.A[n], axes=((0, 1), (0, 1)))
        else:
            T1 = tensordot(F[(n - 1, n)], bra.A[n], axes=(0, 0), conj=(0, 1))
            F[(n, n + 1)] = tensordot(T1, ket.A[n], axes=((0, 1, 2), (0, 1, 2)))


def _update3(n, F, bra, op, ket, to, nr_phys, on_aux):
    if to == 'last':
        if nr_phys == 1:
            F[(n, n + 1)] = ncon([bra.A[n], F[(n - 1, n)], ket.A[n], op.A[n]],
                                        ((4, 5, -1), (4, 2, 1), (1, 3, -3), (2, 5, 3, -2)), conjs=(1, 0, 0, 0))
        elif nr_phys == 2 and not on_aux:
            F[(n, n + 1)] = ncon([bra.A[n], F[(n - 1, n)], ket.A[n], op.A[n]],
                                        ((4, 5, 6, -1), (4, 2, 1), (1, 3, 6, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
        else:  # nr_phys == 2 and on_aux:
            F[(n, n + 1)] = ncon([bra.A[n], F[(n - 1, n)], ket.A[n], op.A[n]],
                                        ((4, 6, 5, -1), (4, 2, 1), (1, 6, 3, -3), (2, 5, 3, -2)), (1, 0, 0, 0))
    elif to == 'first':
        if nr_phys == 1:
            F[(n, n - 1)] = ncon([ket.A[n], F[(n + 1, n)], op.A[n], bra.A[n]],
                                        ((-1, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 5)), (0, 0, 0, 1))
        elif nr_phys == 2 and not on_aux:
            F[(n, n - 1)] = ncon([ket.A[n], F[(n + 1, n)], op.A[n], bra.A[n]],
                                        ((-1, 2, 6, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 4, 6, 5)), (0, 0, 0, 1))
        else:  # nr_phys == 2 and on_aux:
            F[(n, n - 1)] = ncon([ket.A[n], F[(n + 1, n)], op.A[n], bra.A[n]],
                                        ((-1, 6, 2, 1), (1, 3, 5), (-2, 4, 2, 3), (-3, 6, 4, 5)), (0, 0, 0, 1))
