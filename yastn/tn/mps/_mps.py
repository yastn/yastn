""" Mps structure and its basic manipulations. """
from __future__ import annotations
import numbers
from ... import tensor, initialize, YastnError

###################################
#   auxiliary for basic algebra   #
###################################

def Mps(N) -> yastn.tn.mps.MpsMpo:
    r""" Generate empty MPS for system of `N` sites, fixing :code:`nr_phys=1`. """
    return MpsMpo(N, nr_phys=1)


def Mpo(N) -> yastn.tn.mps.MpsMpo:
    r""" Generate empty MPO for system of `N` sites, fixing :code:`nr_phys=2`."""
    return MpsMpo(N, nr_phys=2)


def add(*states, amplitudes=None) -> yastn.tn.mps.MpsMpo:
    r"""
    Linear superposition of several MPS/MPOs with specific amplitudes, i.e., :math:`\sum_j \textrm{amplitudes[j]} \times \textrm{states[j]}`.

    Compression (truncation of bond dimensions) is not performed.

    Parameters
    ----------
    states: Sequence[yastn.tn.mps.MpsMpo]

    amplitudes: Sequence[scalar]
        If :code:`None`, all amplitudes are assumed to be 1.
    """
    if amplitudes is None:
        amplitudes = [1] * len(states)

    if len(states) != len(amplitudes):
        raise YastnError('Number of MpsMPo-s to add must be equal to the number of coefficients in amplitudes.')

    phi = MpsMpo(N=states[0].N, nr_phys=states[0].nr_phys)

    if any(psi.N != phi.N for psi in states):
        raise YastnError('All MpsMpo to add must have equal number of sites.')
    if any(psi.nr_phys != phi.nr_phys for psi in states):
        raise YastnError('All states to add should be either Mps or Mpo.')
    if any(psi.pC != None for psi in states):
        raise YastnError('Absorb central block of MpsMpo-s before calling add.')
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


def multiply(a, b, mode=None) -> yastn.tn.mps.MpsMpo:
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
        a, b: yastn.tn.mps.MpsMpo, yastn.tn.mps.MpsMpo
            a pair of MPO and MPS or two MPO's to be multiplied

        mode: str
           mode for :meth:`Tensor.fuse_legs()<yastn.Tensor.fuse_legs>`; If :code:`None` (default)
           use default setting from YASTN tensor's
           :ref:`configuration<tensor/configuration:yastn configuration>`.
    """
    if a.N != b.N:
        raise YastnError('MpsMpo-s to multiply must have equal number of sites.')

    if a.pC is not None or b.pC is not None:
        raise YastnError('Absorb central blocks of MpsMpo-s before calling multiply.')

    nr_phys = a.nr_phys + b.nr_phys - 2

    if a.nr_phys == 1:
        raise YastnError(' Multiplication by MPS from left is not supported.')

    phi = MpsMpo(N=a.N, nr_phys=nr_phys)

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
    # The basic structure of MPS/MPO with `N` sites.

    def __init__(self, N=1, nr_phys=1):
        r"""
        Initialize empty MPS (:code:`nr_phys=1`) or MPO (:code:`nr_phys=2`)
        for system of `N` sites. Empty MPS/MPO has no tensors assigned.

        MpsMpo tensors (sites) are indexed by integers :math:`0, 1, 2, \ldots, N-1`,
        where :math:`0` corresponds to the `'first'` site.
        They can be accessed with ``[]`` operator.
        MPS/MPO can contain a central block associated with a bond and indexed by a tuple :math:`(n, n+1)`.
        At most one central block is allowed.

        :ref:`The legs' order <mps/properties:index convention>` of each tensor is:
        (0) virtual leg pointing towards the first site, (1) 1st physical leg, i.e., :math:`|\textrm{ket}\rangle`,
        (2) virtual leg pointing towards the last site, and, in case of MPO, (3) 2nd physical leg, i.e., :math:`\langle \textrm{bra}|`.
        """
        if not isinstance(N, numbers.Integral) or N <= 0:
            raise YastnError('Number of Mps sites N should be a positive integer.')
        if nr_phys not in (1, 2):
            raise YastnError('Number of physical legs, nr_phys, should be 1 or 2.')
        self._N = N
        self.A = {i: None for i in range(N)}  # dict of mps tensors; indexed by integers
        self.pC = None  # index of the central block, None if it does not exist
        self._first = 0  # index of the first lattice site
        self._last = N - 1  # index of the last lattice site
        self._nr_phys = nr_phys
        self.factor = 1  # multiplicative factor is real and positive (e.g. norm)

    @property
    def first(self):
        return self._first

    @property
    def last(self):
        return self._last

    @property
    def nr_phys(self):
        return self._nr_phys

    @property
    def N(self):
        return self._N

    def __len__(self):
        return self._N

    @property
    def config(self):
        return self.A[0].config

    def sweep(self, to='last', df=0, dl=0) -> Iterator[int]:
        r"""
        Generator of indices of all sites going from the first site to the last site, or vice-versa.

        Parameters
        ----------
        to: str
            'first' or 'last'.
        df, dl: int
            shift iterator by :math:`df \ge 0` and :math:`dl \ge 0` from the first and the last site, respectively.
        """
        if to == 'last':
            return range(df, self.N - dl)
        if to == 'first':
            return range(self.N - 1 - dl, df - 1, -1)
        raise YastnError('"to" in sweep should be in "first" or "last"')

    def __getitem__(self, n) -> yastn.Tensor:
        """ Return tensor corresponding to n-th site."""
        try:
            return self.A[n]
        except KeyError as e:
            raise YastnError(f"MpsMpo does not have site with index {n}") from e

    def __setitem__(self, n, tensor):
        """
        Assign tensor to n-th site of Mps or Mpo.

        Assigning central block is not supported.
        """
        if not isinstance(n, numbers.Integral) or n < self.first or n > self.last:
            raise YastnError("MpsMpo: n should be an integer in 0, 1, ..., N-1")
        if tensor.ndim != self.nr_phys + 2:
            raise YastnError(f"MpsMpo: Tensor rank should be {self.nr_phys + 2}")
        self.A[n] = tensor

    def shallow_copy(self) -> yastn.tn.mps.MpsMpo:
        r"""
        New instance of :class:`yastn.tn.mps.MpsMpo` pointing to the same tensors as the old one.

        Shallow copy is usually sufficient to retain the old MPS/MPO.
        """
        phi = MpsMpo(N=self.N, nr_phys=self.nr_phys)
        phi.A = dict(self.A)
        phi.pC = self.pC
        phi.factor = self.factor
        return phi

    def clone(self) -> yastn.tn.mps.MpsMpo:
        r"""
        Makes a clone of MPS or MPO by :meth:`cloning<yastn.Tensor.clone>`
        all :class:`yastn.Tensor<yastn.Tensor>`'s into a new and independent :class:`yastn.tn.mps.MpsMpo`.
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.clone()
        # TODO clone factor ?
        return phi

    def copy(self) -> yastn.tn.mps.MpsMpo:
        r"""
        Makes a copy of MPS or MPO by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.mps.MpsMpo`.
        """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.copy()
        # TODO copy factor ?
        return phi

    def conj(self) -> yastn.tn.mps.MpsMpo:
        """ Makes a conjugation of the object. """
        phi = self.shallow_copy()
        for ind, ten in phi.A.items():
            phi.A[ind] = ten.conj()
        return phi

    def transpose(self) -> yastn.tn.mps.MpsMpo:
        """ Transpose of MPO. For MPS, return self. Same as :attr:`self.T<yastn.tn.mps.MpsMpo.T>`"""
        if self.nr_phys == 1:
            return self
        phi = self.shallow_copy()
        for n in phi.sweep(to='last'):
            phi.A[n] = phi.A[n].transpose(axes=(0, 3, 2, 1))
        return phi

    @property
    def T(self) -> yastn.tn.mps.MpsMpo:
        r""" Transpose of MPO. For MPS, return self. Same as :meth:`self.transpose()<yastn.tn.mps.MpsMpo.transpose>` """
        return self.transpose()

    def __mul__(self, multiplier) -> yastn.tn.mps.MpsMpo:
        """ New MPS/MPO with the first tensor multiplied by a scalar. """
        phi = self.shallow_copy()
        am = abs(multiplier)
        if am > 0:
            phi.factor = am * self.factor
            phi.A[0] = phi.A[0] * (multiplier / am)
        else:
            phi.A[0] = phi.A[0] * multiplier
        return phi

    def __rmul__(self, number) -> yastn.tn.mps.MpsMpo:
        """ New MPS/MPO with the first tensor multiplied by a scalar. """
        return self.__mul__(number)

    def __truediv__(self, number) -> yastn.tn.mps.MpsMpo:
        """ Divide MPS/MPO by a scalar. """
        return self.__mul__(1 / number)

    def __add__(self, phi) -> yastn.tn.mps.MpsMpo:
        """ Sum of two Mps's or two Mpo's. """
        return add(self, phi)

    def __sub__(self, phi) -> yastn.tn.mps.MpsMpo:
        """ Subtraction of two Mps's or two Mpo's. """
        return add(self, phi, amplitudes=(1, -1))

    def __matmul__(self, phi) -> yastn.tn.mps.MpsMpo:
        """ Multiply Mpo by Mpo or Mps. """
        return multiply(self, phi)

    def orthogonalize_site(self, n, to='first', normalize=True) -> None:
        r"""
        Performs QR (or RQ) decomposition of on-site tensor at :code:`n`-th position.
        Two modes of usage are

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
            n: int
                index of site to be orthogonalized

            to: str
                a choice of canonical form: :code:`'last'` or :code:`'first'`.

            normalize: bool
                Whether to keep track of the norm by accumulating it in self.factor
                Default is True, i.e., sets the norm to unity.
                The central blocks at the end of the procedure is normalized to unity.
        """
        if self.pC is not None:
            raise YastnError('Only one central block is allowed. Attach the existing central block before orthogonalizing site.')

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
            raise YastnError('"to" should be in "first" or "last"')
        nR = R.norm()
        self.A[self.pC] = R / nR
        self.factor = 1 if normalize else self.factor * nR

    def diagonalize_central(self, opts_svd, normalize=True) -> number:
        r"""
        Perform svd of the central block C = U @ S @ V. Truncation is done based on opts_svd.

        Attach U and V respectively to the left and right sites
        (or to the central site at the edges when there are no left or right sites).

        Returns the norm of truncated Schmidt values, normalized by the norm of all Schmidt values.

        Parameters
        ----------
        opts_svd: dict
            Options passed to :meth:`yastn.linalg.truncation_mask`. It includes information on how to truncate the Schmidt values.

        normalize: bool
            Whether to keep track of the norm of untruncated Schmidt values by accumulating it in self.factor
            Default is True, i.e., sets the norm to unity.
            The truncated Schmidt values (central block) at the end of the procedure is normalized to unity.
        """
        # if self.pC is None:  does not happen now
        #     return 0
        discarded = 0.
        if self.pC is not None:

            U, S, V = tensor.svd(self.A[self.pC], axes=(0, 1), sU=1)
            nSold = S.norm()

            mask = tensor.truncation_mask(S, **opts_svd)
            nSout = tensor.bitwise_not(mask).apply_mask(S, axes=0).norm()
            discarded = nSout / nSold

            U, S, V = mask.apply_mask(U, S, V, axes=(1, 0, 0))
            nS = S.norm()
            self.A[self.pC] = S / nS
            self.factor = 1 if normalize else self.factor * nSold

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

    def remove_central(self) -> None:
        """ Removes (ignores) the central block. Do nothing if is does not exist. """
        if self.pC is not None:
            del self.A[self.pC]
            self.pC = None

    def absorb_central(self, to='last') -> None:
        r"""
        Absorb central block towards the first or the last site.

        If the central block is outside of the chain, it is goes in the one direction possible.
        Do nothing if central does not exist.

        Parameters
        ----------
        to: str
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

    def norm(self) -> number:
        """ Calculate norm of MPS/MPO via canonization. """
        phi = self.shallow_copy()
        #if not phi.is_canonical(to='first'):
        phi.canonize_(to='first', normalize=False)
        return phi.factor

    def canonize_(self, to='first', normalize=True) -> yastn.tn.mps.MpsMpo:
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        (:code:`to='first'` or :code:`to='last'`, respectively)
        using :meth:`QR decomposition<yastn.linalg.qr>`.
        It is assumed that tensors are enumerated
        by index increasing from `0` (:code:`first`) to `N-1` (:code:`last`).

        Finally, the trivial central block is attached to the end of the chain.

        It updates MPS/MPO in place.

        Parameters
        ----------
        to: str
            :code:`'first'` (default) or :code:`'last'`.

        normalize: bool
            Whether to keep track of the norm of initial, untruncated state in self.factor.
            Default is True, i.e. sets the norm to unity.
            The individual tensors at the end of the procedure are in a proper canonical form.
        """
        self.absorb_central(to=to)
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            self.absorb_central(to=to)
        return self

    def is_canonical(self, to='first', n=None, tol=1e-12) -> bool:
        r"""
        Check if MPS/MPO is in a canonical form.

        Parameters
        ----------
        to: str
            :code:`'first'` (default) or code:`'last'`.

        n: int
            Can check a single site if int provided. If None, check all sites.

        tol: float
            Tolerance of the check. Default is 1e-12.
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
        return self.pC is None  # True if no central block

    def truncate_(self, to='last', opts_svd=None, normalize=True) -> number:
        r"""
        Sweep through the MPS/MPO and put it in right/left canonical form
        (:code:`to='first'` or :code:`to='last'`, respectively)
        using :meth:`yastn.linalg.svd_with_truncation` decomposition.
        It is assumed that tensors are enumerated
        by index increasing from 0 (:code:`first`) to N-1 (:code:`last`).

        Access to singular values during sweeping allows truncation of virtual spaces.
        This truncation makes sense if MPS/MPO is already in the canonical form (not checked/enforced)
        in the direction opposite to the current sweep, i.e., right canonical form for :code:`to='last'`
        or left canonical form for :code:`to='first'`.

        The MPS/MPO is updated in place.

        Returns the norm of truncated elements normalized by the norm of the untruncated state.

        Parameters
        ----------
        to: str
            :code:`'last'` (default) or :code:`'first'`.

        normalize: bool
            Whether to keep track of the norm of initial, untruncated state in self.factor.
            Default is True, i.e. sets the norm to unity.
            The individual tensors at the end of the procedure are in a proper canonical form.

        opts_svd: dict
            options passed to :meth:`yastn.linalg.svd_with_truncation`,
            including options governing truncation. Default is {'tol': 1e-13}.
        """
        discarded2_total = 0.
        if opts_svd is None:
            opts_svd = {'tol': 1e-13}
        for n in self.sweep(to=to):
            self.orthogonalize_site(n=n, to=to, normalize=normalize)
            discarded_local = self.diagonalize_central(opts_svd=opts_svd, normalize=normalize)
            discarded2_local = discarded_local ** 2
            discarded2_total = discarded2_local + discarded2_total - discarded2_total * discarded2_local
            self.absorb_central(to=to)
        return self.config.backend.sqrt(discarded2_total)

    def merge_two_sites(self, bd) -> yastn.Tensor:
        r"""
        Merge two neighbouring mps sites and return the resulting tensor.

        Fuse physical indices.

        Parameters
        ----------
        bd: tuple
            (n, n + 1), index of two sites to merge.

        Returns
        -------
        tensor formed from A[n] and A[n + 1]
        """
        nl, nr = bd
        return tensor.tensordot(self.A[nl], self.A[nr], axes=(2, 0))

    def unmerge_two_sites(self, AA, bd, opts_svd) -> number:
        r"""
        Unmerge rank-4 tensor into two neighbouring MPS sites and a central block
        using :meth:`yastn.linalg.svd_with_truncation` to trunctate the bond dimension.

        Input tensor should be a result of :meth:`merge_two_sites` (or fused consistently).

        Parameters
        ----------
        AA: Tensor
            Tensor to be unmerged. It gets unfused in place during the operation.

        bd: tuple
            (n, n + 1), index of two sites to merge.

        Returns
        -------
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

    def get_bond_dimensions(self) -> Sequence[int]:
        r"""
        Returns total bond dimensions of all virtual spaces along MPS/MPO from
        the first to the last site, including "trivial" leftmost and rightmost virtual spaces.
        This gives a tuple with `N + 1` elements.
        """
        Ds = [self.A[n].get_shape(axes=0) for n in self.sweep(to='last')]
        Ds.append(self.A[self.last].get_shape(axes=2))
        return tuple(Ds)

    def get_bond_charges_dimensions(self) -> Sequence[dict[Sequence[int], int]]:
        r"""
        Returns list of charge sectors and their dimensions for all virtual spaces along MPS/MPO
        from the first to the last site, including "trivial" leftmost and rightmost virtual spaces.
        Each element of the list is a dictionary {charge: sectorial bond dimension}.
        This gives a list with `N + 1` elements.
        """
        tDs = []
        for n in self.sweep(to='last'):
            leg = self.A[n].get_legs(axes=0)
            tDs.append(leg.tD)
        leg = self.A[self.last].get_legs(axes=2)
        tDs.append(leg.tD)
        return tDs

    def get_virtual_legs(self) -> Sequence[yastn.Leg]:
        r"""
        Returns :class:`yastn.Leg` of all virtual spaces along MPS/MPO from
        the first to the last site, in the form of the `0th` leg of each MPS/MPO tensor.
        Finally, append the rightmost virtual spaces, i.e., `2nd` leg of the last tensor,
        conjugating it so that all legs have signature `-1`.
        This gives a list with `N + 1` elements.
        """
        legs = [self.A[n].get_legs(axes=0) for n in self.sweep(to='last')]
        legs.append(self.A[self.last].get_legs(axes=2).conj())
        return legs

    def get_physical_legs(self) -> Sequence[yastn.Leg] | Sequence[tuple(yastn.Leg, yastn.Leg)]:
        r"""
        Returns :class:`yastn.Leg` of all physical spaces along MPS/MPO from
        the first to the last site. For MPO return a tuple of ket and bra spaces for each site.
        """
        if self.nr_phys == 2:
            return [self.A[n].get_legs(axes=(1, 3)) for n in self.sweep(to='last')]
        return [self.A[n].get_legs(axes=1) for n in self.sweep(to='last')]

    def get_entropy(self, alpha=1) -> Sequence[number]:
        r"""
        Entropy of bipartition across each bond, along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost cuts.  This gives a list with N + 1 elements.

        Parameters
        ----------
        alpha: int
            Order of Renyi entropy.
            The default value is 1, which corresponds to the Von Neumann entropy.
        """
        schmidt_spectra = self.get_Schmidt_values()
        return [tensor.entropy(spectrum ** 2, alpha=alpha) for spectrum in schmidt_spectra]


    def get_Schmidt_values(self) -> Sequence[yastn.Tensor]:
        r"""
        Schmidt values for bipartition across all bonds along MPS/MPO from the first to the last site,
        including "trivial" leftmost and rightmost cuts. This gives a list with `N + 1` elements.
        Schmidt values are stored as diagonal tensors and are normalized.
        """
        SV = []
        psi = self.shallow_copy()
        psi.canonize_(to='first')
        # canonize_ attaches trivial central block at the end
        axes = (0, (1, 2)) if self.nr_phys == 1 else (0, (1, 2, 3))
        _, sv, _ = tensor.svd(psi.A[self.first], axes=axes, sU=1)
        SV.append(sv)
        for n in psi.sweep(to='last'):
            psi.orthogonalize_site(n=n, to='last')
            _, sv, _ = tensor.svd(psi.A[psi.pC], sU=1)
            SV.append(sv)
            psi.absorb_central(to='last')
        return SV

    def save_to_dict(self) -> dict[str, dict | number]:
        r"""
        Serialize MPS/MPO into a dictionary.

        Each element represents serialized :class:`yastn.Tensor`
        (see, :meth:`yastn.Tensor.save_to_dict`) of the MPS/MPO.
        Absorbs central block if it exists.
        """
        psi = self.shallow_copy()
        psi.absorb_central()  # make sure central block is eliminated
        out_dict = {
            'N': psi.N,
            'nr_phys': psi.nr_phys,
            'factor': psi.factor, #.item(),
            'A': {}
        }
        for n in psi.sweep(to='last'):
            out_dict['A'][n] = psi[n].save_to_dict()
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
        psi = self.shallow_copy()
        psi.absorb_central()  # make sure central block is eliminated
        file.create_dataset(my_address+'/N', data=psi.N)
        file.create_dataset(my_address+'/nr_phys', data=psi.nr_phys)
        file.create_dataset(my_address+'/factor', data=psi.factor)
        for n in self.sweep(to='last'):
            psi[n].save_to_hdf5(file, my_address+'/A/'+str(n))
