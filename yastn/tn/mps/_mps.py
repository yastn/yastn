""" Mps structure and its basic manipulations. """
from __future__ import annotations
import numbers
from ... import tensor, initialize, YastnError

###################################
#   auxiliary for basic algebra   #
###################################

def Mps(N):
    r"""
    Generate empty MPS for system of *N* sites, fixing :code:`nr_phys=1`.

    Parameters
    ----------
    N : int
        number of sites

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    return MpsMpo(N, nr_phys=1)


def Mpo(N):
    r"""
    Generate empty MPO for system of *N* sites, fixing :code:`nr_phys=2`.

    Parameters
    ----------
    N : int
        number of sites

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    return MpsMpo(N, nr_phys=2)


def add(*states, amplitudes=None) -> MpsMpo:
    r"""
    Linear superposition of several MPS/MPOs with specific amplitudes, i.e., :math:`\sum_j \textrm{amplitudes[j]} \times \textrm{states[j]}`.

    Compression (truncation of bond dimensions) is not performed.

    Parameters
    ----------
    states : sequence(yastn.tn.mps.MpsMpo)

    amplitudes : list(scalar)
        If :code:`None`, all amplitudes are assumed to be 1.

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    if amplitudes is None:
        amplitudes = [1] * len(states)

    if len(states) != len(amplitudes):
        raise YastnError('MPS: Number of Mps-s must be equal to the number of coefficients in amplitudes.')

    phi = MpsMpo(N=states[0].N, nr_phys=states[0].nr_phys)

    if any(psi.N != phi.N for psi in states):
        raise YastnError('MPS: All states must have equal number of sites.')
    if any(psi.nr_phys != phi.nr_phys for psi in states):
        raise YastnError('MPS: All states should be either Mps or Mpo.')
    if any(psi.pC != None for psi in states):
        raise YastnError('MPS: Absorb central sites of mps-s before calling add.')
    # legf = states[0][phi.first].get_legs(axes=0)
    # legl = states[0][phi.last].get_legs(axes=2)
    #if any(psi.virtual_leg('first') != legf or psi.virtual_leg('last') != legl for psi in states):
    #    raise YastnError('MPS: Addition')

    amplitudes = [x * psi.factor for x, psi in zip(amplitudes, states)]

    n = phi.first
    d = {(j,): amplitudes[j] * psi.A[n] for j, psi in enumerate(states)}
    common_legs = (0, 1) if phi.nr_phys == 1 else (0, 1, 3)
    phi.A[n] = initialize.block(d, common_legs)

    common_legs = (1,) if phi.nr_phys == 1 else (1, 3)
    for n in phi.sweep(to='last', df=1, dl=1):
        d = {(j, j): psi.A[n] for j, psi in enumerate(states)}
        phi.A[n] = initialize.block(d, common_legs)

    n = phi.last
    d = {(j,): psi.A[n] for j, psi in enumerate(states)}
    common_legs = (1, 2) if phi.nr_phys == 1 else (1, 2, 3)
    phi.A[n] = initialize.block(d, common_legs)

    return phi


def multiply(a, b, mode=None) -> MpsMpo:
    r"""
    Performs MPO-MPS product resulting in a new MPS, or
    MPO-MPO product resulting in a new MPO.

    For MPS/MPO with no symmetry, the bond dimensions
    of the result are given by the product of the bond dimensions of the inputs.
    For symmetric MPS/MPO the virtual spaces of the result are given by
    :ref:`fusion <tensor/algebra:fusion of legs (reshaping)>`
    (i.e. resolving tensor product into direct sum) of the virtual spaces
    of the inputs.

    .. math::
        V^{\textrm{result}}_{j,j+1} = V^{a}_{j,j+1} \otimes V^{b}_{j,j+1}
        = \sum_{c} V^{\textrm{result},c}_{j,j+1},

    where the charge sectors :math:`V^{\textrm{result},c}` are computed during fusion.

    .. note::
        One can equivalently call :code:`a @ b`.

    Parameters
    ----------
        a, b : yastn.tn.mps.MpsMpo, yastn.tn.mps.MpsMpo
            a pair of MPO and MPS or two MPO's to be multiplied

        mode : str
           mode for :meth:`Tensor.fuse_legs()<yastn.Tensor.fuse_legs>`; If :code:`None` (default)
           use default setting from YASTN tensor's
           :ref:`configuration<tensor/configuration:yastn configuration>`.

    Returns
    -------
        yastn.tn.mps.MpsMpo
    """
    if a.N != b.N:
        YastnError('MPS: Mps-s must have equal number of sites.')

    nr_phys = a.nr_phys + b.nr_phys - 2

    if a.nr_phys == 1:
        YastnError('MPS: First argument has to be an MPO.')
    phi = MpsMpo(N=a.N, nr_phys=nr_phys)

    if b.N != a.N:
        raise YastnError('MPS: a and b must have equal number of sites.')
    if a.pC is not None or b.pC is not None:
        raise YastnError('MPS: Absorb central sites of mps-s before calling multiply.')

    axes_fuse = ((0, 3), 1, (2, 4)) if b.nr_phys == 1 else ((0, 3), 1, (2, 4), 5)
    for n in phi.sweep():
        phi.A[n] = tensor.tensordot(a.A[n], b.A[n], axes=(3, 1)).fuse_legs(axes_fuse, mode)
    phi.A[phi.first] = phi.A[phi.first].drop_leg_history(axes=0)
    phi.A[phi.last] = phi.A[phi.last].drop_leg_history(axes=2)
    phi.factor = a.factor * b.factor
    return phi

###################################
#     basic operations on MPS     #
###################################

class MpsMpo:
    r"""
    The basic structure of MPS/MPO with *N* sites.

    MpsMpo tensors (sites) are accessed with usual ``[]`` operator. They are indexed
    by integers from :math:`0, 1, 2, 3, \ldots, N-1` with :math:`0` corresponding to the first site.
    A central block, associated with a bond, is indexed using ordered tuple (n, n+1).
    At most one central block is allowed.

    The :ref:`Index convention<mps/properties:index convention>` for
    legs of each tensor is: virtual leg in the direction of first site, 1st physical leg (:math:`|\textrm{ket}\rangle`),
    virtual leg in the direction of last site, and 2nd physical leg (:math:`\langle \textrm{bra}|`) in case of MPO.
    """

    def __init__(self, N=1, nr_phys=1):
        r"""
        Create empty MPS (:code:`nr_phys=1`) or MPO (:code:`nr_phys=2`)
        for system of *N* sites. Empty MPS/MPO has no tensors assigned.

        Parameters
        ----------
        N : int
            number of sites
        nr_phys : int
            number of physical legs: 1 for MPS (default); 2 for MPO;
        """
        if not isinstance(N, numbers.Integral) or N <= 0:
            raise YastnError('MPS: Number of Mps sites N should be a positive integer.')
        if nr_phys not in (1, 2):
            raise YastnError('MPS: Number of physical legs, nr_phys, should be equal to 1 or 2.')
        self.N = N
        self.A = {i: None for i in range(N)}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central site, None if it does not exist
        self.first = 0  # index of the first lattice site
        self.last = N - 1  # index of the last lattice site
        self.nr_phys = nr_phys
        self.factor = 1  # multiplicative factor is real and positive (e.g. norm)

    @property
    def config(self):
        return self.A[0].config

    def sweep(self, to='last', df=0, dl=0):
        r"""
        Generator of indices of all sites going from the first site to the last site, or vice-versa.

        Parameters
        ----------
        to : str
            'first' or 'last'.
        df, dl : int
            shift iterator by df >= 0 and dl >= 0 from the first and the last site, respectively.
        """
        if to == 'last':
            return range(df, self.N - dl)
        if to == 'first':
            return range(self.N - 1 - dl, df - 1, -1)
        raise YastnError('MPS: Argument "to" should be in ("first", "last")')

    def __getitem__(self, n):
        """ Return tensor corresponding to n-th site."""
        return self.A[n]

    def __setitem__(self, n, tensor):
        """ Assign tensor to n-th site of Mps or Mpo. """
        if not isinstance(n, numbers.Integral) or n < self.first or n > self.last:
            raise YastnError('MPS: n should be an integer in [0, N - 1].')
        if tensor.ndim != self.nr_phys + 2:
            raise YastnError('MPS: Tensor rank should be {}.'.format(self.nr_phys + 2))
        self.A[n] = tensor

    def shallow_copy(self):
        r"""
        New instance of :class:`yastn.tn.mps.MpsMpo` pointing to the same tensors as the old one.

        Shallow copy is usually sufficient to retain the old MPS/MPO.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        phi = MpsMpo(N=self.N, nr_phys=self.nr_phys)
        phi.A = dict(self.A)
        phi.pC = self.pC
        phi.factor = self.factor
        return phi

    def clone(self):
        r"""
        Makes a clone of MPS or MPO by :meth:`cloning<yastn.Tensor.clone>`
        all :class:`yastn.Tensor<yastn.Tensor>`'s into a new and independent :class:`yastn.tn.mps.MpsMpo`.

        .. note::
            Cloning preserves autograd tracking on all tensors.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.clone()
        # TODO clone factor ?
        return phi

    def copy(self):
        r"""
        Makes a copy of MPS or MPO by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.mps.MpsMpo`.

        .. warning::
            this operation does not preserve autograd on the returned :code:`yastn.tn.mps.MpsMpo`.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.copy()
        # TODO copy factor ?
        return phi

    def conj(self):
        r"""
        Makes a conjugation of the object.

        Returns
        -------
        out : conjugated Mps or Mpo, independent of self
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.conj()
        return phi

    def __mul__(self, multiplier):
        """
        New MPS/MPO with first tensor multiplied by a scalar.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        phi = self.shallow_copy()
        am = abs(multiplier)
        if am > 0:
            phi.factor = am * self.factor
            phi.A[0] = phi.A[0] * (multiplier / am)
        else:
            phi.A[0] = phi.A[0] * multiplier
        return phi

    def __rmul__(self, number):
        """
        New MPS/MPO with first tensor multiplied by a scalar.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        return self.__mul__(number)

    def __add__(self, phi):
        """
        Sum of two Mps's or two Mpo's.

        Parameters
        ----------
        mps : Mps or Mpo (same as self)

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        return add(self, phi)

    def __sub__(self, phi):
        """
        Subtraction of two Mps's or two Mpo's.

        Parameters
        ----------
        mps : Mps or Mpo (same as self)

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        return add(self, phi, amplitudes=(1, -1))

    def __matmul__(self, phi):
        """
        Multiply Mpo by Mpo or Mps.

        Parameters
        ----------
        multiplier : Mps or Mpo

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        return multiply(self, phi)

    def orthogonalize_site(self, n, to='first', normalize=True):
        r"""
        Performs QR (or RQ) decomposition of on-site tensor at :code:`n`-th position.
        Two typical modes of usage are

            * ``to='last'`` - Advance left canonical form: Assuming first `n - 1` sites are already
              in the left canonical form, brings n-th site to left canonical form,
              i.e., extends left canonical form by one site towards :code:`'last'` site::

                 --A---            --
                |  |    = Identity|
                 --A*--            --

            * ``to='first'`` - Advance right canonical form: Assuming all `m > n` sites are already
              in right canonical form, brings n-th site to right canonical form, i.e.,
              extends right canonical form by one site towards :code:`'first'` site::

                --A---    --
                  |   | =   |Identity
                --A*--    --

        Parameters
        ----------
            n : int
                index of site to be orthogonalized

            to : str
                a choice of canonical form: :code:`'last'` or :code:`'first'`.

            normalize : bool
                If :code:`True`, central block is normalized to unity according
                to standard 2-norm, and :code:`self.factor` is set to `1`. If false, the norm
                gets accumulated in :code:`self.factor`.
        """
        if self.pC is not None:
            raise YastnError('MPS: Only one central block is possible. Attach the existing central block first.')

        if to == 'first':
            self.pC = (n - 1, n)
            ax = (1, 2) if self.nr_phys == 1 else (1, 2, 3)
            # A = ax(A)--Q--0 1--R--0(0A) => 0--Q*--    --
            #                                   1   2 =   |Identity
            #                                0--Q---    --
            self.A[n], R = self.A[n].qr(axes=(ax, 0), sQ=-1, Qaxis=0, Raxis=1)
        elif to == 'last':
            self.pC = (n, n + 1)
            ax = (0, 1) if self.nr_phys == 1 else (0, 1, 3)
            # A = ax(A)--Q--2 0--R--1(2A) =>  --Q*--2            --
            #                                0  1     = Identity|
            #                                 --Q---2            --
            self.A[n], R = self.A[n].qr(axes=(ax, 2), sQ=1, Qaxis=2)
        else:
            raise YastnError('MPS: Argument "to" should be in ("first", "last")')
        nR = R.norm()
        self.A[self.pC] = R / nR
        self.factor = 1 if normalize else self.factor * nR

    def diagonalize_central(self, opts_svd, normalize=True):
        r"""
        Perform svd of the central site C = U @ S @ V. Truncation can be done based on opts_svd.

        Attach U and V respective to the left and right sites.

        Parameters
        ----------
        opts_svd : dict
            Options passed for svd. iIncludes information how to truncate bond.

        normalize : bool
            If true, S is normalized to 1 according to the standard 2-norm.

        Return
        ------
        discarded : float
            norm of discarded singular values normalized by the remining ones
        """
        if self.pC is not None:
            U, S, V = tensor.svd(self.A[self.pC], axes=(0, 1), sU=1)

            mask = tensor.truncation_mask(S, **opts_svd)
            discarded = tensor.bitwise_not(mask).apply_mask(S, axes=0).norm() / S.norm()

            U, S, V = mask.apply_mask(U, S, V, axes=(1, 0, 0))
            self.A[self.pC] = S / S.norm() if normalize else S
            n1, n2 = self.pC

            if n1 >= self.first:
                ax = (-0, -1, 1) if self.nr_phys == 1 else (-0, -1, 1, -3)
                self.A[n1] = tensor.ncon([self.A[n1], U], (ax, (1, -2)))
            else:
                self.A[self.pC] = U @ self.A[self.pC]

            if n2 <= self.last:
                self.A[n2] = V @ self.A[n2]
            else:
                self.A[self.pC] = self.A[self.pC] @ V

            return discarded
        return 0

    def remove_central(self):
        r"""
        Removes (ignores) the central site. Do nothing if is does not exist.
        """
        if self.pC is not None:
            del self.A[self.pC]
            self.pC = None

    def absorb_central(self, to='last'):
        r"""
        Absorb central site towards the first or the last site.

        If the central site is outside of the chain, it is goes in the one direction possible.
        Do nothing if central does not exist.

        Parameters
        ----------
        to : str
            'last' or 'first'.
        """
        if self.pC is not None:
            C = self.A.pop(self.pC)
            n1, n2 = self.pC
            self.pC = None

            if (to == 'first' and n1 >= self.first) or n2 > self.last:
                ax = (-0, -1, 1) if self.nr_phys == 1 else (-0, -1, 1, -3)
                self.A[n1] = tensor.ncon([self.A[n1], C], (ax, (1, -2)))
            else:  # (to == 'last' and n2 <= self.last) or n1 < self.first
                self.A[n2] = C @ self.A[n2]

    def norm(self):
        r""" Calculate norm of MPS/MPO via canonization.

        Returns
        -------
        scalar
        """
        phi = self.shallow_copy()
        #if not phi.is_canonical(to='first'):
        phi.canonize_(to='first', normalize=False)
        return phi.factor

    def canonize_(self, to='first', normalize=True):
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        using :meth:`QR<yastn.linalg.qr>` decomposition by setting
        :code:`to='first'` or :code:`to='last'`. It is assumed that tensors are enumerated
        by index increasing from `0` (:code:`first`) to `N-1` (:code:`last`).

        Finally, the trivial central block is attached to the end of the chain.

        Parameters
        ----------
        to : str
            :code:`'first'` (default) or :code:`'last'`.

        normalize : bool
            If :code:`true` (default), the central block and thus MPS/MPO is normalized
            to unity according to the standard 2-norm. If false, the norm gets accumulated in :code:`self.factor`.

        Returns
        -------
        self : yastn.tn.mps.MpsMpo
            in-place canonized MPS/MPO
        """
        self.absorb_central(to=to)
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            self.absorb_central(to=to)
        return self

    def is_canonical(self, to='first', n=None, tol=1e-12):
        r"""
        Check if MPS/MPO is in a canonical form.

        Parameters
        ----------
        to : str
            :code:`'first'` (default) or code:`'last'`.

        n : int
            Can check a single site if int provided. If None, check all sites.

        tol : float
            Tolerance of the check. Default is 1e-12.

        Returns
        -------
        bool
        """
        if to == 'first':
            # 0--A*--
            #    1   2
            # 0--A---
            cl = (1, 2) if self.nr_phys == 1 else (1, 2, 3)
        else: # to == 'last':
            #  --A*--2
            # 0  1
            #  --A---2
            cl = (0, 1) if self.nr_phys == 1 else (0, 1, 3)
        it = self.sweep(to=to) if n is None else [n]
        for n in it:
            x = tensor.tensordot(self.A[n], self.A[n].conj(), axes=(cl, cl))
            x = x.drop_leg_history()
            x0 = initialize.eye(config=x.config, legs=x.get_legs((0, 1)))
            if (x - x0.diag()).norm() > tol:  # == 0
                return False
        return self.pC is None  # True if no central site

    def truncate_(self, to='last', normalize=True, opts_svd=None):
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        using :meth:`yastn.linalg.svd_with_truncation` decomposition by setting
        :code:`to='first'` or :code:`to='last'`. It is assumed that tensors are enumerated
        by index increasing from 0 (:code:`first`) to N-1 (:code:`last`).

        Access to singular values during sweeping allows to truncate virtual spaces.
        This truncation makes sense if MPS/MPO is already in the canonical form
        in the direction opposite to current sweep, i.e., left canonical form for :code:`to='last'`
        or right canonical form for :code:`to='first'`.

        The MPS/MPO is canonized in-place.

        Parameters
        ----------
        to : str
            :code:`'last'` (default) or :code:`'first'`.

        normalize : bool
            If :code:`true` (default), the central block and thus MPS/MPO is normalized
            to unity according to the standard 2-norm.

        opts_svd : dict
            options passed to :meth:`yastn.linalg.svd_with_truncation`,
            including options governing truncation.

        Returns
        -------
        scalar
            maximal norm of the discarded singular values after normalization
        """
        discarded_max = 0.
        if opts_svd is None:
            opts_svd = {}
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            discarded = self.diagonalize_central(opts_svd=opts_svd, normalize=normalize)
            discarded_max = max(discarded_max, discarded)
            self.absorb_central(to=to)
        return discarded_max

    def merge_two_sites(self, bd):
        r"""
        Merge two neighbouring mps sites and return the resulting tensor.

        Fuse physical indices.

        Parameters
        ----------
        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        ----------
        yastn.Tensor
            tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        return tensor.tensordot(self.A[nl], self.A[nr], axes=(2, 0))

    def unmerge_two_sites(self, AA, bd, opts_svd):
        r"""
        Unmerge rank-4 tensor into two neighbouring MPS sites and a central block
        using :meth:`yastn.linalg.svd_with_truncation` to trunctate the bond dimension.

        Input tensor should be a result of :meth:`merge_two_sites` (or fused consistently).

        Parameters
        ----------
        AA : Tensor
            Tensor to be unmerged. It gets unfused in place during the operation.

        bd : tuple
            (n, n + 1), index of two sites to merge.

        Returns
        -------
        scalar
            normalized discarded weight :math:`\sum_{i\in\textrm{discarded}}\lambda_i/\sum_i\lambda_i`,
            where :math:`\lambda_i` are singular values across the bond.
        """
        nl, nr = bd
        # axes = (1,) if self.nr_phys == 1 else (1, 3)
        # AA = AA.unfuse_legs(axes=axes)
        # axes = ((0, 1), (2, 3)) if self.nr_phys == 1 else ((0, 1, 4), (2, 3, 5))
        axes = ((0, 1), (2, 3)) if self.nr_phys == 1 else ((0, 1, 2), (3, 4, 5))
        self.pC = bd
        U, S, V = tensor.svd(AA, axes=axes, sU=1, Uaxis=2)
        mask = tensor.truncation_mask(S, **opts_svd)
        self.A[nl], self.A[bd], self.A[nr] = mask.apply_mask(U, S, V, axes=(2, 0, 0))
        return tensor.bitwise_not(mask).apply_mask(S, axes=0).norm() / S.norm()  # discarded weight

    def virtual_leg(self, ind):
        if ind == 'first':
            return self.A[self.first].get_legs(axes=0)
        if ind == 'last':
            return self.A[self.last].get_legs(axes=2)

    def get_bond_dimensions(self) -> list[int]:
        r"""
        Returns total bond dimensions of all virtual spaces along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost virtual spaces.
        """
        Ds = [self.A[n].get_shape(axes=0) for n in self.sweep(to='last')]
        Ds.append(self.A[self.last].get_shape(axes=2))
        return tuple(Ds)

    def get_bond_charges_dimensions(self) -> list[dict[tuple[int], int]]:
        r"""
        Returns list of charge sectors and their dimensions for all virtual spaces along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost virtual spaces.
        Each element of the list is a dictionary {charge: sectorial bond dimension}.

        Returns
        -------
        list(dict(tuple(int),int))
        """
        tDs = []
        for n in self.sweep(to='last'):
            leg = self.A[n].get_legs(axes=0)
            tDs.append(leg.tD)
        leg = self.A[self.last].get_legs(axes=2)
        tDs.append(leg.tD)
        return tDs

    def get_entropy(self, alpha=1):
        r"""
        Entropy of bipartition across each bond, along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost cuts.

        Parameters
        ----------
        alpha : int
            value 1 (default) computes Von Neumann entropy.
            Higher values instead compute order `alpha` Renyi entropies.

        Returns
        -------
        list(scalar)
        """
        Entropy = [0] * (self.N + 1)
        psi = self.shallow_copy()
        psi.absorb_central(to='first')
        psi.canonize_(to='last', normalize=False)
        for n in psi.sweep(to='first'):
            psi.orthogonalize_site(n=n, to='first', normalize=False)
            Entropy[n], _, _ = tensor.entropy(psi.A[psi.pC], alpha=alpha)
            psi.absorb_central(to='first')
        return Entropy

    def get_Schmidt_values(self):
        r"""
        Schmidt values for bipartition across all bonds (i-1, i). Schmidt values are stored as diagonal tensors.

        Returns
        -------
        dict((int, int), yastn.Tensor)
        """
        SV = {}
        psi = self.shallow_copy()
        psi.canonize_(to='last', normalize=False)
        psi.absorb_central(to='first')
        for n in psi.sweep(to='first', df=1):
            psi.orthogonalize_site(n=n, to='first', normalize=False)
            _, sv, _ = tensor.svd(psi.A[psi.pC], sU=1)
            SV[psi.pC] = sv
            psi.absorb_central(to='first')
        return SV

    def save_to_dict(self):
        r"""
        Serialize MPS/MPO into a dictionary.

        Each element represents serialized :class:`yastn.Tensor`
        (see :meth:`yastn.Tensor.save_to_dict`)
        of the MPS/MPO starting from the first site to the last.

        Returns
        -------
        dict(int, dict)
        """
        out_dict = {
            'nr_phys': self.nr_phys,
            'sym': {
                'SYM_ID': self[0].config.sym.SYM_ID,
                'NSYM': self[0].config.sym.NSYM
            },
            'A' : {}
        }
        for n in self.sweep(to='last'):
            out_dict['A'][n] = self.A[n].save_to_dict()
        return out_dict

    def save_to_hdf5(self, file, my_address):
        r"""
        Save MPS/MPO into a HDF5 file.

        Parameters
        ----------
        file: File
            A `pointer` to a file opened by the user

        my_address: str
            Name of a group in the file, where the Mps will be saved, e.g., 'state/'
        """
        file.create_dataset(my_address+'/nr_phys', data=self.nr_phys)
        file.create_dataset(my_address+'/sym/SYM_ID', data=self[0].config.sym.SYM_ID)
        file.create_dataset(my_address+'/sym/NSYM', data=self[0].config.sym.NSYM)
        for n in self.sweep(to='last'):
            self.A[n].save_to_hdf5(file, my_address+'/A/'+str(n))
