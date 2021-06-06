""" Environments for the <mps| mpo |mps> and <mps|mps>  contractions. """
from yast import ncon, match_legs, tensordot, expmv, vdot
from ._mps import YampsError


def measure_overlap(bra, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env2(bra=bra, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


def measure_mpo(bra, op, ket):
    r"""
    Calculate overlap <bra|ket>

    Returns
    -------
    overlap : float or complex
    """
    env = Env3(bra=bra, op=op, ket=ket)
    env.setup(to='first')
    return env.measure(bd=(-1, 0))


class _EnvParent:
    def __init__(self, bra=None, ket=None, orthogonal=None) -> None:
        self.ket = ket
        self.bra = bra if bra is not None else ket
        self.N = ket.N
        self.nr_phys = ket.nr_phys
        self.nr_layers = 2
        self.F = {}  # dict for environments
        self.ortho = [] if orthogonal is None else orthogonal
        self.nr_ortho = len(self.ortho)
        self.Fortho = [{} for _ in range(self.nr_ortho)]
        self.Aortho = []

        if self.bra.nr_phys != self.ket.nr_phys:
            raise YampsError('bra and ket should have the same number of physical legs.')
        if self.bra.N != self.ket.N:
            raise YampsError('bra and ket should have the same number of sites.')

        ll = self.N - 1
        for ii in range(self.nr_ortho):
            self.Fortho[ii][(-1, 0)] = match_legs(tensors=[self.ortho[ii].A[0], self.ket.A[0]],
                                            legs=[self.ortho[ii].left[0], self.ket.left[0]],
                                            conjs=[1, 0], val='ones')
            self.Fortho[ii][(ll + 1, ll)] = match_legs(tensors=[self.ket.A[ll], self.ortho[ii].A[ll]],
                                            legs=[self.ket.right[0], self.ortho[ii].right[0]],
                                            conjs=[0, 1], val='ones')

    def setup(self, to='last'):
        r"""
        Setup all environments in the direction given by to.

        Parameters
        ----------
        to : str
            'first' or 'last'.
        """
        for n in self.ket.sweep(to=to):
            self.update(n, to=to)


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

        It is equall to the overlap <bra|op|ket> up to the contribution from n-th site of bra.

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


    def update(self, n, to='last'):
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
        for ii in range(self.nr_ortho):
            _update2(n, self.Fortho[ii], self.ortho[ii], self.ket, to, self.nr_phys)


    def update_Aortho(self, n):
        """ Update projection of states to be orthogonal to on psi. """
        self.Aortho = []
        for ii in range(self.nr_ortho):
            T1 = tensordot(self.Fortho[ii][(n - 1, n)], self.ortho[ii].A[n], axes=(0, 0), conj=(0, 1))
            self.Aortho.append(tensordot(T1, self.Fortho[ii][(n + 1, n)], axes=(self.nr_phys + 1, 1)))


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
        self.F[(-1, 0)] = match_legs(tensors=[self.bra.A[0], self.ket.A[0]],
                                        legs=[self.bra.left[0], self.ket.left[0]],
                                        conjs=[1, 0], val='ones')
        # right boundary
        ll = self.N - 1
        self.F[(ll + 1, ll)] = match_legs(tensors=[self.ket.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.bra.right[0]],
                                        conjs=[0, 1], val='ones')


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

    def __init__(self, bra=None, op=None, ket=None, on_aux=False, orthogonal=None):
        r"""
        Initialize structure for :math:`\langle {\rm bra} | {\rm opp} | {\rm ket} \rangle` related operations.

        Parameters
        ----------
        bra : mps
            mps for :math:`| {\rm bra} \rangle`. If None, it is the same as ket.
        opp : mps
            mps for opperator opp.
        ket : mps
            mps for :math:`| {\rm ket} \rangle`.
        """
        super().__init__(bra, ket, orthogonal)
        self.op = op
        self.nr_layers = 3
        self.on_aux = on_aux
        self._expmv_ncv = {}
        if self.op.N != self.N:
            raise YampsError('op should should have the same number of sites as ket.')

        # left boundary
        self.F[(-1, 0)] = match_legs(tensors=[self.bra.A[0], self.op.A[0], self.ket.A[0]],
                                        legs=[self.bra.left[0], self.op.left[0], self.ket.left[0]],
                                        conjs=[1, 0, 0], val='ones')
        # right boundary
        ll = self.N - 1
        self.F[(ll + 1, ll)] = match_legs(tensors=[self.ket.A[ll], self.op.A[ll], self.bra.A[ll]],
                                        legs=[self.ket.right[0], self.op.right[0], self.bra.right[0]],
                                        conjs=[0, 0, 1], val='ones')


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
        if self.nr_phys == 1:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
        elif self.nr_phys == 2 and not self.on_aux:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        else:
            T1 = ncon([self.F[(nl, n)], A, self.op.A[n], self.F[(nr, n)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        for ii in range(self.nr_ortho):
            x = vdot(self.Aortho[ii], T1)
            T1 = T1.apxb(self.Aortho[ii], -x)
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

        if not hasattr(self, 'op_merged'):
            self.op_merged = {}
        if bd not in self.op_merged:
            OO = tensordot(self.op.A[n1], self.op.A[n2], axes=(3, 0))
            OO = OO.fuse_legs(axes=(0, (1, 3), (2, 4), 5))
            self.op_merged[bd] = OO
        OO = self.op_merged[bd]

        if self.nr_phys == 1:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, 4), (2, -2, 3, 5), (4, 5, -3)), (0, 0, 0, 0))
        if self.nr_phys == 2 and not self.on_aux:
            return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                        ((-1, 2, 1), (1, 3, -3, 4), (2, -2, 3, 5), (4, 5, -4)), (0, 0, 0, 0))
        return ncon([self.F[(nl, n1)], AA, OO, self.F[(nr, n2)]],
                    ((-1, 2, 1), (1, -2, 3, 4), (2, -3, 3, 5), (4, 5, -4)), (0, 0, 0, 0))


    def update_A(self, n, dt, opts):
        """ Updates env.psi.A[n] by exp(dt Heff1). """
        if n in self._expmv_ncv:
            opts['ncv'] = self._expmv_ncv[n]
        f = lambda x: self.Heff1(x, n)
        self.ket.A[n], info = expmv(f, self.ket.A[n], dt, **opts, normalize=True, return_info=True)
        self._expmv_ncv[n] = info['ncv']


    def update_C(self, dt, opts):
        """ Updates env.psi.A[bd] by exp(dt Heff0). """
        bd = self.ket.pC
        if bd[0] != -1 and bd[1] != self.N:  # do not update central sites outsite of the chain
            if bd in self._expmv_ncv:
                opts['ncv'] = self._expmv_ncv[bd]
            f = lambda x: self.Heff0(x, bd)
            self.ket.A[bd], info = expmv(f, self.ket.A[bd], dt, **opts, normalize=True, return_info=True)
            self._expmv_ncv[bd] = info['ncv']


    def update_AA(self, bd, dt, opts, opts_svd):
        """ Merge two sites given in bd into AA, updates AA by exp(dt Heff2) and unmerge the sites. """
        ibd = bd[::-1]
        if ibd in self._expmv_ncv:
            opts['ncv'] = self._expmv_ncv[ibd]
        AA = self.ket.merge_two_sites(bd)
        f = lambda v: self.Heff2(v, bd)
        AA, info = expmv(f, AA, dt, **opts, normalize=True, return_info=True)
        self._expmv_ncv[ibd] = info['ncv']
        self.ket.unmerge_two_sites(AA, bd, opts_svd)


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
