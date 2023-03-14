import numpy as np
from typing import NamedTuple
from ... import ones, rand, ncon, Leg, random_leg, YastError
from ...operators import Qdit
from ._mps import Mpo, Mps, add
from ._latex2term import latex2term, GeneratorError


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
    operators : tuple(yast.Tensor)
        local operators in the product that are different than the identity.
        *i*-th operator is acting at position :code:`positions[i]` 
    """
    amplitude : float = 1.0
    positions : tuple = ()
    operators : tuple = ()


def generate_single_mpo(I, term):   # this can be private
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
    """
    single_mpo = I.copy()
    for site, op in zip(term.positions[::-1], term.operators[::-1]):
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
    single_mpo[0] = term.amplitude * single_mpo[0]
    return single_mpo


def generate_mpo(I, terms, normalize=False, opts=None, packet=50):  # can use better algorithm to compress
    """
    Generate MPO provided a list of :class:`Hterm`-s and identity MPO `I`.

    If the number of MPOs is large, adding them all together can result 
    in large intermediate MPO. By specifying ``packet`` size, the groups of MPO-s 
    are truncated at intermediate steps before continuing with summing. 

    Parameters
    ----------
    term: list of :class:`Hterm`
        product operators making up the MPO
    I: yast.Tensor
        on-site identity operator
    normalize: bool
        True if the result should be normalized
    opts: dict
        options for truncation of the result
    packet: int
        how many ``Hterm``s (MPOs of bond dimension 1) should be truncated at once
    """
    ip, M_tot, Nterms = 0, None, len(terms)
    while ip < Nterms:
        H1s = [generate_single_mpo(I, terms[j]) for j in range(ip, min([Nterms, ip + packet]))]
        M = add(*H1s)
        M.truncate_(to='first', opts=opts, normalize=normalize)
        ip += packet
        if not M_tot:
            M_tot = M.copy()
        else:
            M_tot = M_tot + M
            M_tot.truncate_(to='first', opts=opts, normalize=normalize)
    return M_tot

def generate_single_mps(term, N):  # obsolate - DELETE  (not docummented)
    r"""
    Generate an MPS given vectors for each site in the MPS.

    Parameters
    ----------
    term: :class:`Hterm`
        instruction to create the Mps which is a product of
        operators element.operator at location element.position
        and with amplitude element.amplitude.
    N: int
        MPS size
    """
    if len(term.positions) != len(set(term.positions)):
        raise GeneratorError("List contains more than one operator for a single position.\n \
            Multiplication of two vectors is not defined.")
    if len(set(term.positions)) != N:
        raise GeneratorError("Provide term for each site in MPS.")
    single_mps = Mps(N)
    for n in range(N):
        if n in term.positions:
            op = term.operators[term.positions == n]
        else:
            raise GeneratorError("Provide term for each site in MPS.")
        single_mps.A[n] = op.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    return term.amplitude * single_mps

def generate_mps(terms, N, normalize=False, opts=None, packet=50):   #  DELETE
    """
    Generate MPS provided a list of :class:`Hterm`-s.

    If the number of MPSs is large, adding them all together can result 
    in large intermediate MPS. By specifying ``packet`` size, the groups of MPO-s 
    are truncated at intermediate steps before continuing with summing.
    
    Parameters
    ----------
    N: int
       number of sites
    term: list of :class:`Hterm`
        product operators making up the MPS
    normalize: bool
        True if the result should be normalized
    opts: dict
        options for truncation of the result
    packet: int
        how many single MPO-s of bond dimension 1 shuold be truncated at ones
    """
    ip, M_tot, Nterms = 0, None, len(terms)
    while ip < Nterms:
        H1s = [generate_single_mps(terms[j], N) for j in range(ip, min([Nterms, ip + packet]))]
        M = add(*H1s)
        M.truncate_(to='first', opts=opts, normalize=normalize)
        ip += packet
        if not M_tot:
            M_tot = M.copy()
        else:
            M_tot = M_tot + M
            M_tot.truncate_(to='first', opts=opts, normalize=normalize)
    return M_tot


class Generator:
    
    def __init__(self, N, operators, map=None, Is=None, parameters=None, opts={"tol": 1e-14}):
        """
        Generator allowing creation of MPO/MPS from a set of local operators.

        Generated MPO tensors have following signature::

                   (-1) (physical bra)
                     |
            (+1)--|MPO_i|--(-1)
                     |
                   (+1) (physical ket)

        Parameters
        ----------
        N : int
            number of sites of MPS/MPO.
        operators : object
            a set of local operators, e.g., an instance of :class:`yast.operators.Spin12`.
            The ``operators`` object is expected to provide a map :code:`operators.to_dict()`
            from string labels to operators (:class:`yast.Tensor`).
        map : dict(int,int)
            custom permutation of N sites indexed from 0 to N-1 , ``{3: 0, 21: 1, ...}``,
            with values ordered as 0, 1, ..., N - 1.
            If ``None``, assumes no permutation, i.e., :code:`{site: site for site in range(N)}`.
        Is : dict(int,str)
            For each site, specify identity operator by providing its string label, i.e., ``{0: 'I', 1: 'I', ...}``.
            If ``None``, uses default specification ``{i: 'I' for i in range(N)}``.
        parameters : dict
            Default parameters used by the interpreters :meth:`Generator.mpo` and :meth:`Generator.mps`.
            If None, uses default ``{'sites': [*map.keys()]}``.
        opts : dict
            used if compression is needed. Options passed to :meth:`yast.linalg.svd`.
        
        Notes
        ------
        * Names `minus` and `1j` are reserved paramters in self.parameters

        * Write operator `a` on site `3` as `a_{3}`.
        
        * Write element if matrix `A` with indicies `(1,3)` as `A_{1,2}`.
        
        * Write sumation over one index `j` taking values from 1D-array `listA` as `\sum_{j \in listA}`.
        
        * Write sumation over indicies `j0,j1` taking values from 2D-array `listA` as `\sum_{j0,j1 \in listA}`.
        
        * In an expression only round brackets, i.e., ().
        """
        self.N = N
        self._ops = operators
        self._map = {i: i for i in range(N)} if map is None else map
        if len(self._map) != N or sorted(self._map.values()) != list(range(N)):
            raise YastError("MPS: Map is inconsistent with mps of N sites.")
        self._Is = {k: 'I' for k in self._map.keys()} if Is is None else Is
        if self._Is.keys() != self._map.keys():
            raise YastError("MPS: Is is inconsistent with map.")
        if not all(hasattr(self._ops, v) and callable(getattr(self._ops, v)) for v in self._Is.values()):
            raise YastError("MPS: operators do not contain identity specified in Is.")

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
        """
        Generate random seed number for random number generator used in backend of self.config

        Parameters
        ----------
        seed : int
            Seed number for random number generator.
        """
        self.config.backend.random_seed(seed)

    def I(self):
        """ Returns identity MPO. """
        return self._I.copy()

    def random_mps(self, n=None, D_total=8, sigma=1, dtype='float64'):
        """
        Generate a random Mps of total charge n and virtual bond dimension D_total.

        Parameters
        ----------
        n : int
            total charge
        D_total : int
            total dimension of virtual space
        sigma : int
            variance of Normal distribution from which dimensions of charge sectors
            are drawn.
        dtype : string
            number format, e.g., 'float64'
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
        raise YastError("MPS: Random mps is a zero state. Check parameters, or try running again in this is due to randomness of the initialization. ")

    def random_mpo(self, D_total=8, sigma=1, dtype='float64'):
        """
        Generate a random MPO of virtual bond dimension D_total.

        Mainly, for testing.

        Parameters
        ----------
        D_total : int
            total dimension of virtual space
        sigma : int
            variance of Normal distribution from which dimensions of charge sectors
            are drawn.
        dtype : string
            number format, e.g., 'float64'
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
        raise YastError("Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization).")

    def mps_from_latex(self, psi_str, vectors=None, parameters=None):
        r"""
        Generate simple mps form the instruction in psi_str.

        Parameters
        ----------
        psi_str : str
            instruction in latex-like format
        vectors : dict
            dictionary with vectors for the generator. All should be given as
            a dictionary with elements in a format:
            name : lambda j: tensor
                where 
                name - is a name of an element which can be used in psi_str,
                j - single index for lambda function,
                tensor - is a yast.Tensor with one physical index.
        parameters : dict
            dictionary with parameters for the generator

        Returns
        --------
            :class:`yamps.Mps`
        """
        parameters = {**self.parameters, **parameters}
        c2 = latex2term(psi_str, parameters)
        c3 = self._term2Hterm(c2, vectors, parameters)
        return generate_mps(c3, self.N)

    def mps_from_templete(self, templete, vectors=None, parameters=None):
        r"""
        Convert instruction in a form of single_term-s to yamps MPO.

        single_term is a templete which which take named from operators and templetes.

        Parameters
        -----------
        templete: list
            List of single_term objects. The object is defined in ._latex2term
        vectors : dict
            dictionary with vectors for the generator. All should be given as
            a dictionary with elements in a format:
            name : lambda j: tensor
                where 
                name - is a name of an element which can be used in psi_str,
                j - single index for lambda function,
                tensor - is a yast.Tensor with one physical index.
        parameters: dict
            Keys for the dict define the expressions that occur in H_str

        Returns
        --------
            :class:`yamps.Mps`
        """
        parameters = {**self.parameters, **parameters}
        c3 = self._term2Hterm(templete, vectors, parameters)
        return generate_mps(c3, self.N)

    def mpo_from_latex(self, H_str, parameters=None):   
        r"""
        Convert latex-like string to yamps MPO.

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
        --------
            :class:`yamps.Mpo`
        """
        parameters = {**self.parameters, **parameters}
        c2 = latex2term(H_str, parameters)
        c3 = self._term2Hterm(c2, self._ops.to_dict(), parameters)
        return generate_mpo(self._I, c3)
    
    def mpo_from_templete(self, templete, parameters=None):   # remove from docs (DELETE)
        r"""
        Convert instruction in a form of single_term-s to yamps MPO.

        single_term is a templete which which take named from operators and templetes.

        Parameters
        -----------
        templete: list
            List of single_term objects. The object is defined in ._latex2term
        parameters: dict
            Keys for the dict define the expressions that occur in H_str

        Returns
        --------
            :class:`yamps.Mpo`
        """
        parameters = {**self.parameters, **parameters}
        c3 = self._term2Hterm(templete, self._ops.to_dict(), parameters)
        return generate_mpo(self._I, c3)

    def _term2Hterm(self, c2, obj_yast, obj_number):
        """
        Helper function to rewrite the instruction given as a list of single_term-s (see _latex2term)
        to a list of Hterm-s (see here).

        Differentiates operators from numberical values.

        Parameters
        ----------
        c2 : list
            list of single_term-s
        obj_yast : dict
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
                    mapindex = self._map[indicies[0]] if len(indicies) == 1 else YastError("Operator has to have single index as defined by self._map")
                    positions.append(mapindex) 
                    operators.append(obj_yast[element](mapindex))
                else:
                    # the only other option is that is a number, imaginary number is in self.obj_number
                    amplitude *= float(element)
            Hterm_list.append(Hterm(amplitude, positions, operators))
        return Hterm_list

def random_dense_mps(N, D, d, **kwargs):
    """Generate random mps with physical dimension d and virtual dimension D."""
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mps(D_total=D)

def random_dense_mpo(N, D, d, **kwargs):
    """Generate random mpo with physical dimension d and virtual dimension D."""
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mpo(D_total=D)
