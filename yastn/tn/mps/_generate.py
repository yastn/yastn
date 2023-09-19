import numpy as np
import numbers
from typing import NamedTuple
from ... import ones, rand, zeros, ncon, Leg, random_leg, YastnError, Tensor, block, svd_with_truncation
from ...operators import Qdit
from ._mps import Mpo, Mps
from ._latex2term import latex2term


class Hterm(NamedTuple):
    r"""
    Defines a product operator :math:`O\in\mathcal{A}(\mathcal{H})` on product Hilbert space :math:`\mathcal{H}=\otimes_i \mathcal{H}_i`
    where :math:`\mathcal{H}_i` is a local Hilbert space of *i*-th degree of freedom.
    The product operator :math:`O = \otimes_i o_i` is a tensor product of local operators :math:`o_i`.
    Unless explicitly specified, the :math:`o_i` is assumed to be an identity operator.

    If operators are fermionic, execution of swap gates enforces fermionic order (the last operator in the list acts first).

    Parameters
    ----------
    amplitude : number
        numerical multiplier in front of the operator product.
    positions : tuple(int)
        positions of the local operators :math:`o_i` in the product different than identity
    operators : tuple(yastn.Tensor)
        local operators in the product that are different than the identity.
        *i*-th operator is acting at position :code:`positions[i]`
    """
    amplitude : float = 1.0
    positions : tuple = ()
    operators : tuple = ()


def generate_product_mpo(I, term, amplitude=True):
    r"""
    Apply local operators specified by term in :class:`Hterm` to the MPO `I`.

    MPO `I` is presumed to be an identity.
    Apply swap_gates to introduce fermionic degrees of freedom
    (fermionic order is the same as order of sites in Mps).
    With this respect, local operators specified in term.operators are applied starting with the last element,
    i.e., from right to left.

    Parameters
    ----------
    term: :class:`Hterm`

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    single_mpo = I.copy()
    for site, op in zip(term.positions[::-1], term.operators[::-1]):
        if site < 0 or site > I.N or not isinstance(site, numbers.Integral):
            raise YastnError("position in Hterm should be in 0, 1, ..., N-1 ")
        if not op.s == (1, -1):
            raise YastnError("operator in Hterm should be a matrix with signature (1, -1)")
        op = op.add_leg(axis=0, s=-1)
        leg = op.get_legs(axes=0)
        one = ones(config=op.config, legs=(leg, leg.conj()))
        temp = ncon([op, single_mpo[site]], [(-1, -2, 1), (-0, 1, -3, -4)])
        single_mpo[site] = temp.fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
        for n in range(site):
            temp = ncon([single_mpo[n], one], [(-0, -2, -3, -5), (-1, -4)])
            temp = temp.swap_gate(axes=(1, 2))
            single_mpo[n] = temp.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')
    for n in single_mpo.sweep():
        single_mpo[n] = single_mpo[n].drop_leg_history(axes=(0, 2))
    if amplitude:
        single_mpo[0] = term.amplitude * single_mpo[0]
    return single_mpo


class GenerateMpoTemplate(NamedTuple):
    config: NamedTuple = None
    basis : list = None
    trans : list = None
    tleft : list = None


def generate_mpo_template(I, terms, return_amplitudes=False):
    r"""
    Precompute an amplitude-independent template that is then used to generate MPO by :meth:`yastn.tn.mps.generate_mpo_fast`

    Parameters
    ----------
    term: list of :class:`Hterm`
        product operators making up the MPO
    I: yastn.tn.mps.MpsMpo
        identity MPO
    return_amplitudes: bool
        Apart from template, return also amplitudes = [term.amplitude for term in terms]

    Returns
    -------
    template [NamedTuple] or (template [NamedTuple], amplitude [list])
    """
    H1s = [generate_product_mpo(I, term, amplitude=False) for term in terms]
    cfg = H1s[0][0].config
    mapH = np.zeros((len(H1s), I.N), dtype=int)

    basis, t1bs, t2bs, tfbs, ifbs = [], [], [], [], []
    for n in I.sweep():
        base = []
        for m, H1 in enumerate(H1s):
            ind = next((ind for ind, v in enumerate(base) if (v - H1[n]).norm() < 1e-13), None)  # site-tensors differing by less then 1e-13 are considered identical
            if ind is None:
                ind = len(base)
                base.append(H1[n])
            mapH[m, n] = ind

        t1bs.append([ten.get_legs(axes=0).t[0] for ten in base])
        t2bs.append([ten.get_legs(axes=2).t[0] for ten in base])
        base = [ten.fuse_legs(axes=((0, 2), 1, 3)).drop_leg_history() for ten in base]
        tfb = [ten.get_legs(axes=0).t[0] for ten in base]
        tfbs.append(tfb)
        ifbs.append([sum(x == y for x in tfb[:i]) for i, y in enumerate(tfb)])
        base = block(dict(enumerate(base)), common_legs=(1, 2)).drop_leg_history()
        basis.append(base)

    tleft = [t1bs[0][i] for i in mapH[:, 0]]

    trans = []
    for n in I.sweep():
        mapH0 = mapH[:, 0]
        mapH, rind, iind = np.unique(mapH[:, 1:], axis=0, return_index=True, return_inverse=True)

        i2bs = {t: {} for t in t2bs[n]}
        for ii, rr in enumerate(rind):
            i2bs[t2bs[n][mapH0[rr]]][ii] = len(i2bs[t2bs[n][mapH0[rr]]])

        i1bs = {t: 0 for t in t1bs[n]}
        for ii, rr in enumerate(mapH0):
            i1bs[t1bs[n][rr]] += 1

        leg1 = Leg(cfg, s=-1, t=list(i1bs.keys()), D=list(i1bs.values()))
        leg2 = basis[n].get_legs(axes=0).conj()
        leg3 = Leg(cfg, s=1, t=list(i2bs.keys()), D=[len(x) for x in i2bs.values()])
        tran = zeros(config=cfg, legs=[leg1, leg2, leg3])

        li = {x: -1 for x in t1bs[n]}
        for bl, br in zip(mapH0, iind):
            lt = t1bs[n][bl]
            li[lt] += 1
            ft = tfbs[n][bl]
            fi = ifbs[n][bl]
            rt = t2bs[n][bl]
            ri = i2bs[rt][br]
            tran[lt + ft + rt][li[lt], fi, ri] += 1
        trans.append(tran)

    template = GenerateMpoTemplate(config=cfg, basis=basis, trans=trans, tleft=tleft)
    if return_amplitudes:
        amplitudes = [term.amplitude for term in terms]
        return template, amplitudes
    return template


def generate_mpo_fast(template, amplitudes, opts=None):
    r"""
    Fast generation of MPO representing the lists(Hterm) that differ only in amplitudes.

    Some precomputations in :meth:`yastn.tn.mps.generate_mpo` might be slow.
    When only amplitudes in Hterms are changing (e.g. for time-dependent Hamiltonian),
    MPO generation can be significantly speeded up by precalculating an reusing amplitude-independent `template`.
    The latter is done with :meth:`yastn.tn.mps.generate_mpo_template`.

    Parameters
    ----------
    template: NamedTuple
        calculated with :meth:`yastn.tn.mps.generate_mpo_template`
    amplidutes: list(numbers)
        list of amplitudes that would appear in :class:`Hterm`.
        The order of the list should match the order of Hterms supplemented to :meth:`yastn.tn.mps.generate_mpo_template`.
    opts: dict
        The generator function employs svd while compressing MPO bond dimension.
        opts allows passing options to :meth:`yastn.linalg.svd_with_truncation`
        Default None sets truncation `tol` close to the numerical precision, which should effectively result in lossless compression.

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    if opts is None:
        opts = {'tol': 1e-13}

    Js = {}
    for a, t in zip(amplitudes, template.tleft):
        if t in Js:
            Js[t].append(a)
        else:
            Js[t] = [a]

    dtype = 'complex128' if any(isinstance(a, complex) for a in amplitudes) else template.config.default_dtype
    J = Tensor(config=template.config, s=(-1, 1), dtype=dtype)
    for t, val in Js.items():
        J.set_block(ts=(t, t), Ds=(1, len(val)), val=val)

    M = Mpo(len(template.basis))
    for n in M.sweep():
        #   nJ = J @ template.trans[n]
        #   if n < M.last:
        #       nJ, S, V = svd_with_truncation(nJ, axes=((0, 1), 2), sU=1, **opts)
        #       J = S @ V
        #   M[n] = ncon([nJ, template.basis[n]], [[0, 1, -2], [1, -1, -3]])
        nJ = J @ template.trans[n]
        nJ = ncon([nJ, template.basis[n]], [[0, 1, -3], [1, -1, -2]])
        if n < M.last:
            nJ, S, V = svd_with_truncation(nJ, axes=((0, 1, 2), 3), sU=1, **opts)
            nS = S.norm()
            nJ = nS * nJ
            J = (S / nS) @ V
        M[n] = nJ.transpose(axes=(0, 1, 3, 2))
    return M


def generate_mpo(I, terms, opts=None):
    r"""
    Generate MPO provided a list of :class:`Hterm`\-s and identity MPO `I`.

    It is a shorthand for :meth:`yastn.tn.mps.generate_mpo_template` and :meth:`yastn.tn.mps.generate_mpo_fast`,
    but without storying the template that helps generate MPOs for different amplitudes in front of product operators.

    Parameters
    ----------
    term: list of :class:`Hterm`
        product operators making up the MPO
    I: yastn.tn.mps.MpsMpo
        identity MPO
    opts: dict
        generator employs svd while compressing MPO bond dimension.
        opts allows passing options to :meth:`yastn.linalg.svd_with_truncation`
        Default None sets truncation `tol` close to the numerical precision, which should result in effectively lossless compression.

    Returns
    -------
    yastn.tn.mps.MpsMpo
    """
    template, amplitudes = generate_mpo_template(I, terms, return_amplitudes=True)
    return generate_mpo_fast(template, amplitudes, opts=opts)


def generate_product_mps(vectors):
    r"""
    Generate an MPS given vectors for each site in the MPS.

    Parameters
    ----------
    vectors: list[yastn.Tensor]
        vectors will be attributed to consecuative MPS sites.
        Each vector should have signature s=+1.
        They can have non-zero charge, that will be converted into virtual legs.
        The size of MPS is follows as len(vectors).
    """
    psi = Mps(len(vectors))
    rt = (0,) * vectors[0].config.sym.NSYM
    for n, vec in zip(psi.sweep(to='first'), vectors[::-1]):
        if not vec.s == (1,):
            raise YastnError("Vector should have s = (1,).")
        psi[n] = vec.add_leg(axis=1, s=1, t=rt).add_leg(axis=0, s=-1)
        rt = psi[n].get_legs(axes=0).t[0]
    return psi


class Generator:

    def __init__(self, N, operators, map=None, Is=None, parameters=None, opts={"tol": 1e-13}):
        r"""
        Generator is a convenience class building MPOs from a set of local operators.
        Generated MPO have following :ref:`index order<mps/properties:index convention>` and signature::

                     3 (-1) (physical bra)
                     |
            (-1) 0--|MPO_i|--2 (+1)
                     |
                     1 (+1) (physical ket)

        Parameters
        ----------
        N : int
            number of sites of MPS/MPO.
        operators : object or dict(str,yastn.Tensor)
            a set of local operators, e.g., an instance of :class:`yastn.operators.Spin12`.
            Or a dictionary with string-labeled local operators, including at least ``{'I': <identity operator>,...}``.
        map : dict(int,int)
            custom labels of N sites indexed from 0 to N-1 , ``{3: 0, 21: 1, ...}``.
            If ``None``, the sites are labled as :code:`{site: site for site in range(N)}`.
        Is : dict(int,str)
            For each site (using default or custom label), specify identity operator by providing
            its string key as defined in ``operators``.
            If ``None``, assumes ``{i: 'I' for i in range(N)}``, which is compatible with all predefined
            ``operators``.
        parameters : dict
            Default parameters used by the interpreters :meth:`Generator.mpo` and :meth:`Generator.mps`.
            If None, uses default ``{'sites': [*map.keys()]}``.
        opts : dict
            used if compression is needed. Options passed to :meth:`yastn.linalg.svd_with_truncation`.
        """
        # Notes
        # ------
        # * Names `minus` and `1j` are reserved paramters in self.parameters
        # * Write operator `a` on site `3` as `a_{3}`.
        # * Write element if matrix `A` with indicies `(1,3)` as `A_{1,2}`.
        # * Write sumation over one index `j` taking values from 1D-array `listA` as `\sum_{j \in listA}`.
        # * Write sumation over indicies `j0,j1` taking values from 2D-array `listA` as `\sum_{j0,j1 \in listA}`.
        # * In an expression only round brackets, i.e., ().
        self.N = N
        self._ops = operators
        self._map = {i: i for i in range(N)} if map is None else map
        if len(self._map) != N or sorted(self._map.values()) != list(range(N)):
            raise YastnError("MPS: Map is inconsistent with mps of N sites.")
        self._Is = {k: 'I' for k in self._map.keys()} if Is is None else Is
        if self._Is.keys() != self._map.keys():
            raise YastnError("MPS: Is is inconsistent with map.")
        if not all(hasattr(self._ops, v) and callable(getattr(self._ops, v)) for v in self._Is.values()):
            raise YastnError("MPS: operators do not contain identity specified in Is.")

        self._I = Mpo(self.N)
        for label, site in self._map.items():
            local_I = getattr(self._ops, self._Is[label])
            self._I.A[site] = local_I().add_leg(axis=0, s=-1).add_leg(axis=2, s=1)

        self.config = self._I.A[0].config
        self.parameters = {} if parameters is None else parameters
        self.parameters["minus"] = -float(1.0)
        self.parameters["1j"] = 1j

        self.opts = opts

    def random_seed(self, seed):
        r"""
        Set seed for random number generator used in backend (of self.config).

        Parameters
        ----------
        seed : int
            Seed number for random number generator.
        """
        self.config.backend.random_seed(seed)

    def I(self):
        r"""
        Indetity MPO derived from identity in local operators class.

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        return self._I.copy()

    def random_mps(self, n=None, D_total=8, sigma=1, dtype='float64'):
        r"""
        Generate a random MPS of total charge ``n`` and bond dimension ``D_total``.

        Parameters
        ----------
        n : int
            total charge
        D_total : int
            largest bond dimension
        sigma : int
            standard deviation of the normal distribution from which dimensions of charge sectors
            are drawn.
        dtype : string
            number format, e.g., ``'float64'`` or ``'complex128'``

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        if n is None:
            n = (0,) * self.config.sym.NSYM
        try:
            n = tuple(n)
        except TypeError:
            n = (n,)
        an = np.array(n, dtype=int)

        psi = Mps(self.N)

        lr = Leg(self.config, s=1, t=(tuple(an * 0),), D=(1,),)
        for site in psi.sweep(to='first'):
            lp = self._I[site].get_legs(axes=1)
            nl = tuple(an * (self.N - site) / self.N)
            if site != psi.first:
                ll = random_leg(self.config, s=-1, n=nl, D_total=D_total, sigma=sigma, legs=[lp, lr])
            else:
                ll = Leg(self.config, s=-1, t=(n,), D=(1,),)
            psi.A[site] = rand(self.config, legs=[ll, lp, lr], dtype=dtype)
            lr = psi.A[site].get_legs(axes=0).conj()
        if sum(lr.D) == 1:
            return psi
        raise YastnError("MPS: Random mps is a zero state. Check parameters, or try running again in this is due to randomness of the initialization. ")

    def random_mpo(self, D_total=8, sigma=1, dtype='float64'):
        r"""
        Generate a random MPO with bond dimension ``D_total``.

        Parameters
        ----------
        D_total : int
            largest bond dimension
        sigma : int
            standard deviation of the normal distribution from which dimensions of charge sectors
            are drawn.
        dtype : string
            number format, e.g., ``'float64'`` or ``'complex128'``

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        n0 = (0,) * self.config.sym.NSYM
        psi = Mpo(self.N)

        lr = Leg(self.config, s=1, t=(n0,), D=(1,),)
        for site in psi.sweep(to='first'):
            lp = self._I[site].get_legs(axes=1)
            if site != psi.first:
                ll = random_leg(self.config, s=-1, n=n0, D_total=D_total, sigma=sigma, legs=[lp, lr, lp.conj()])
            else:
                ll = Leg(self.config, s=-1, t=(n0,), D=(1,),)
            psi.A[site] = rand(self.config, legs=[ll, lp, lr, lp.conj()], dtype=dtype)
            lr = psi.A[site].get_legs(axes=0).conj()
        if sum(lr.D) == 1:
            return psi
        raise YastnError("Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization).")

    # def mps_from_latex(self, psi_str, vectors=None, parameters=None):
    #     r"""
    #     Generate simple mps form the instruction in psi_str.

    #     Parameters
    #     ----------
    #     psi_str : str
    #         instruction in latex-like format
    #     vectors : dict
    #         dictionary with vectors for the generator. All should be given as
    #         a dictionary with elements in a format:
    #         name : lambda j: tensor
    #             where
    #             name - is a name of an element which can be used in psi_str,
    #             j - single index for lambda function,
    #             tensor - is a yastn.Tensor with one physical index.
    #     parameters : dict
    #         dictionary with parameters for the generator

    #     Returns
    #     -------
    #     yastn.tn.mps.MpsMpo
    #     """
    #     parameters = {**self.parameters, **parameters}
    #     c2 = latex2term(psi_str, parameters)
    #     c3 = self._term2Hterm(c2, vectors, parameters)
    #     return generate_mps(c3, self.N)

    # def mps_from_templete(self, templete, vectors=None, parameters=None):
    #     r"""
    #     Convert instruction in a form of single_term-s to yastn.tn.mps MPO.

    #     single_term is a templete which which take named from operators and templetes.

    #     Parameters
    #     -----------
    #     templete: list
    #         List of single_term objects. The object is defined in ._latex2term
    #     vectors : dict
    #         dictionary with vectors for the generator. All should be given as
    #         a dictionary with elements in a format:
    #         name : lambda j: tensor
    #             where
    #             name - is a name of an element which can be used in psi_str,
    #             j - single index for lambda function,
    #             tensor - is a yastn.Tensor with one physical index.
    #     parameters: dict
    #         Keys for the dict define the expressions that occur in H_str

    #     Returns
    #     -------
    #     yastn.tn.mps.MpsMpo
    #     """
    #     parameters = {**self.parameters, **parameters}
    #     c3 = self._term2Hterm(templete, vectors, parameters)
    #     return generate_mps(c3, self.N)

    def mpo_from_latex(self, H_str, parameters=None, opts=None):
        r"""
        Convert latex-like string to yastn.tn.mps MPO.

        Parameters
        -----------
        H_str: str
            The definition of the MPO given as latex expression. The definition uses string names of the operators given in. The assignment of the location is
            given e.g. for 'cp' operator as 'cp_{j}' (always with {}-brackets!) for 'cp' operator acting on site 'j'.
            The space and * are interpreted as multiplication by a number of by an operator. E.g., to multiply by a number use 'g * cp_j c_{j+1}' where 'g' has to be defines in 'parameters' or writen directly as a number,
            You can define automatic summation with expression '\sum_{j \in A}', where A has to be iterable, one-dimensional object with specified values of 'j'.
        parameters: dict
            Keys for the dict define the expressions that occur in H_str

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        parameters = {**self.parameters, **parameters}
        c2 = latex2term(H_str, parameters)
        c3 = self._term2Hterm(c2, self._ops.to_dict(), parameters)
        if opts is None:
            opts={'tol': 5e-15}
        return generate_mpo(self._I, c3, opts)

    def mpo_from_templete(self, templete, parameters=None):   # remove from docs (DELETE)
        r"""
        Convert instruction in a form of single_term-s to yastn.tn.mps MPO.

        single_term is a templete which which take named from operators and templetes.

        Parameters
        -----------
        templete: list
            List of single_term objects. The object is defined in ._latex2term
        parameters: dict
            Keys for the dict define the expressions that occur in H_str

        Returns
        -------
        yastn.tn.mps.MpsMpo
        """
        parameters = {**self.parameters, **parameters}
        c3 = self._term2Hterm(templete, self._ops.to_dict(), parameters)
        return generate_mpo(self._I, c3)

    def _term2Hterm(self, c2, obj_yast, obj_number):
        r"""
        Helper function to rewrite the instruction given as a list of single_term-s (see _latex2term)
        to a list of Hterm-s (see here).

        Differentiates operators from numberical values.

        Parameters
        ----------
        c2 : list
            list of single_term-s
        obj_yastn : dict
            dictionary with operators for the generator
        obj_number : dict
            dictionary with parameters for the generator
        """
        # can be used with latex-form interpreter or alone.
        Hterm_list = []
        for ic in c2:
            # create a single Hterm using single_term
            amplitude, positions, operators = float(1), [], []
            for iop in ic.op:
                element, *indicies = iop
                if element in obj_number:
                    # can have many indicies for cross terms
                    mapindex = tuple([self._map[ind] for ind in indicies]) if indicies else None
                    amplitude *= obj_number[element] if mapindex is None else obj_number[element][mapindex]
                elif element in obj_yast:
                    # is always a single index for each site
                    mapindex = self._map[indicies[0]] if len(indicies) == 1 else YastnError("Operator has to have single index as defined by self._map")
                    positions.append(mapindex)
                    operators.append(obj_yast[element](mapindex))
                else:
                    # the only other option is that is a number, imaginary number is in self.obj_number
                    amplitude *= float(element)
            Hterm_list.append(Hterm(amplitude, positions, operators))
        return Hterm_list

def random_dense_mps(N, D, d, **kwargs):
    r"""Generate random mps with physical dimension d and virtual dimension D."""
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mps(D_total=D)

def random_dense_mpo(N, D, d, **kwargs):
    r"""Generate random mpo with physical dimension d and virtual dimension D."""
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mpo(D_total=D)
