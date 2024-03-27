from __future__ import annotations
import numpy as np
from ... import rand, Leg, random_leg, YastnError
from ... import load_from_dict as load_from_dict_tensor
from ... import load_from_hdf5 as load_from_hdf5_tensor
from ._mps_obc import Mpo, Mps, MpsMpoOBC
from ...operators import Qdit


def product_mps(vectors, N=None) -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Generate an MPS with bond-dimension one from a list of vectors which get assigned to consecutive MPS sites.

    In `N` is provided, vectors are cyclicly iterated to fill in `N` MPS sites.

    Parameters
    ----------
    vectors: Sequence[yastn.Tensor] | yastn.Tensor
        Tensors will be attributed to consecutive MPS sites.
        Each tensor should have `ndim=1` and the signature `s=+1`.
        They can have non-zero charges that will be converted into matching MPS virtual legs.

    N: Optional[int]
        number of MPS sites. By default, it is equal to the number of provided `vectors`.
    """
    return _product_MpsMpoOBC(vectors, N=N, nr_phys=1)


def product_mpo(operators, N=None) -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Generate an MPO with bond-dimension one from a list of operators which get assigned to consecutive MPO sites.

    In `N` is provided, operators are cyclicly iterated to fill in `N` MPO sites.

    Parameters
    ----------
    operators: Sequence[yastn.Tensor] | yastn.Tensor
        Tensors will be attributed to consecutive MPS sites.
        Each tensor should have `ndim=2` and the signature `s=(+1, -1)`.
        They can have non-zero charges, that will be converted into matching MPO virtual legs.

    N: Optional[int]
        number of MPO sites. By default, it is equal to the number of provided `operators`.

    Example
    -------

    ::

        # This function can help set up an identity MPO,
        # which is the base ingredient for a few other functions
        # generating more complicated MPOs and MPSs.

        import yastn
        import yastn.tn.mps as mps

        ops = yastn.operators.Spin12(sym='Z2')
        I = mps.product_mpo(ops.I(), N=8)

        # Here, each site has the same local physical Hilbert space
        # of dimension 2, consistent with predefined spin-1/2 operators.
        # The MPO I uniquely identifies those local Hilbert spaces.
    """
    return _product_MpsMpoOBC(operators, N=N, nr_phys=2)


def _product_MpsMpoOBC(vectors, N=None, nr_phys=1) -> yastn.tn.mps.MpsMpoOBC:
    """ handles product mpo and mps"""
    try:  # handle inputing single bare Tensor
        vectors = list(vectors)
    except TypeError:
        vectors = [vectors]

    if N is None:
        N = len(vectors)

    psi = MpsMpoOBC(N=N, nr_phys=nr_phys)

    if nr_phys == 1 and any(vec.ndim != 1 for vec in vectors):
        raise YastnError("Vector should have ndim = 1.")
    if nr_phys == 2 and any(vec.ndim != 2 for vec in vectors):
        raise YastnError("Operator should have ndim = 2.")

    Nv = len(vectors)
    if Nv != N:
        vectors = [vectors[n % Nv] for n in psi.sweep(to='last')]

    rt = (0,) * vectors[0].config.sym.NSYM
    for n, vec in zip(psi.sweep(to='first'), vectors[::-1]):
        vec = vec.add_leg(axis=1, s=1, t=rt)
        rt = vec.n
        psi[n] = vec.add_leg(axis=0, s=-1)
    return psi


def random_mps(I, n=None, D_total=8, sigma=1, dtype='float64') -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Generate a random MPS of total charge ``n`` and bond dimension ``D_total``.

    Local Hilbert spaces are read from ket spaces
    of provided MPS or MPO `I`. For instance, `I` can be an identity MPO.
    The number of sites and Tensor config is also inherited from `I`.

    Parameters
    ----------
    I: yastn.tn.mps.MpsMpoOBC
        MPS or MPO that defines local Hilbert spaces.
    n: int
        Total charge of MPS.
        Virtual MPS spaces are drawn randomly from a normal distribution,
        whose mean value changes linearly along the chain from `n` to 0.
    D_total: int
        Largest bond dimension. Due to the random and local nature of the procedure,
        the desired total bond dimension might not be reached on some bonds,
        in particular, for higher symmetries.
    sigma: int
        The standard deviation of the normal distribution.
    dtype: string
        Number format, i.e., ``'float64'`` or ``'complex128'``

    Example
    -------

    ::

        import yastn
        import yastn.tn.mps as mps

        ops = yastn.operators.SpinlessFermions(sym='U1')
        I = mps.product_mpo(ops.I(), N=13)
        psi = mps.random_mps(I, n=6, D_total=8)

        # Random MPS with 13 sites occupied by 6 fermions (fixed by U1 symmetry),
        # and maximal bond dimension 8.
    """
    if n is None:
        n = (0,) * I.config.sym.NSYM
    try:
        n = tuple(n)
    except TypeError:
        n = (n,)
    an = np.array(n, dtype=int)

    psi = Mps(I.N)
    config = I.config

    lr = Leg(config, s=1, t=(tuple(an * 0),), D=(1,),)
    for site in psi.sweep(to='first'):
        lp = I[site].get_legs(axes=1)  # ket leg of MPS/MPO
        nl = tuple(an * (I.N - site) / I.N)  # mean n changes linearly along the chain
        if site != psi.first:
            ll = random_leg(config, s=-1, n=nl, D_total=D_total, sigma=sigma, legs=[lp, lr])
        else:
            ll = Leg(config, s=-1, t=(n,), D=(1,),)
        psi.A[site] = rand(config, legs=[ll, lp, lr], dtype=dtype)
        lr = psi.A[site].get_legs(axes=0).conj()
    if sum(lr.D) == 1:
        return psi
    raise YastnError("MPS: Random mps is a zero state. Check parameters, or try running again in this is due to randomness of the initialization. ")


def random_mpo(I, D_total=8, sigma=1, dtype='float64') -> yastn.tn.mps.MpsMpoOBC:
    r"""
    Generate a random MPO with bond dimension ``D_total``.

    The number of sites and local bra and ket spaces of MPO follow
    from provided MPO `I`, e.g., an identity MPO.
    `I` can be an MPS, in which case its ket spaces are used.

    Parameters
    ----------
    I: yastn.tn.mps.MpsMpoOBC
        MPS or MPO that defines local spaces.
    D_total: int
        Largest bond dimension. Due to the random and local nature of the procedure,
        the desired total bond dimension might not be reached on some bonds,
        in particular, for higher symmetries.
    sigma: int
        Standard deviation of a normal distribution
        from which dimensions of charge sectors are drawn.
    dtype: string
        number format, i.e., ``'float64'`` or ``'complex128'``
    """
    config = I.config
    n0 = (0,) * config.sym.NSYM
    psi = Mpo(I.N)

    lr = Leg(config, s=1, t=(n0,), D=(1,),)
    for site in psi.sweep(to='first'):
        lp = I[site].get_legs(axes=1)
        lpc = I[site].get_legs(axes=3) if I[site].ndim == 4 else lp.conj()
        if site != psi.first:
            ll = random_leg(config, s=-1, n=n0, D_total=D_total, sigma=sigma, legs=[lp, lr, lpc])
        else:
            ll = Leg(config, s=-1, t=(n0,), D=(1,),)
        psi.A[site] = rand(config, legs=[ll, lp, lr, lpc], dtype=dtype)
        lr = psi.A[site].get_legs(axes=0).conj()
    if sum(lr.D) == 1:
        return psi
    raise YastnError("Random mpo is a zero state. Check parameters, or try running again in this is due to randomness of the initialization.")


def random_dense_mps(N, D, d, **kwargs) -> yastn.tn.mps.MpsMpoOBC:
    r"""Generate random MPS with physical dimension d and virtual dimension D."""
    I = product_mpo(Qdit(d=d, **kwargs).I(), N)
    return random_mps(I, D_total=D)


def random_dense_mpo(N, D, d, **kwargs) -> yastn.tn.mps.MpsMpoOBC:
    r"""Generate random MPO with physical dimension d and virtual dimension D."""
    I = product_mpo(Qdit(d=d, **kwargs).I(), N)
    return random_mpo(I, D_total=D)


def load_from_dict(config, in_dict) -> yastn.tn.mps.MpsMpo:
    r"""
    Create MPS/MPO from dictionary.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    in_dict: dict
        dictionary containing serialized MPS/MPO, i.e.,
        a result of :meth:`yastn.tn.mps.MpsMpo.save_to_dict`.
    """
    nr_phys = in_dict['nr_phys']
    N = in_dict['N'] if 'N' in in_dict else len(in_dict['A'])  # backwards compability
    out_mps = MpsMpoOBC(N, nr_phys=nr_phys)
    if 'factor' in in_dict:  # backwards compability
        out_mps.factor = in_dict['factor']
    for n in range(out_mps.N):
        out_mps.A[n] = load_from_dict_tensor(config=config, d=in_dict['A'][n])
    return out_mps


def load_from_hdf5(config, file, my_address) -> yastn.tn.mps.MpsMpo:
    r"""
    Create MPS/MPO from HDF5 file.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    file: File
        A `pointer` to a file opened by the user

    my_address: str
        Name of a group in the file, where the Mps is saved, e.g., './state/'
    """

    nr_phys = int(file[my_address].get('nr_phys')[()])
    N = file[my_address].get('N')
    N = len(file[my_address+'/A'].keys()) if N is None else int(N[()])
    out_Mps = MpsMpoOBC(N, nr_phys=nr_phys)

    factor = file[my_address].get('factor')
    if factor:
        out_Mps.factor = factor[()]
    for n in range(out_Mps.N):
        out_Mps.A[n] = load_from_hdf5_tensor(config, file, my_address+'/A/'+str(n))
    return out_Mps
