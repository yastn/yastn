import numpy as np
from typing import NamedTuple
from ... import ones, rand, ncon, Leg, random_leg, Tensor
from ...operators import Qdit
from ._mps import Mpo, Mps, YampsError, add

from .latex2Hterm import latex2single_term

class Hterm(NamedTuple):
    r"""
    Defines a product operator :math:`O\in\A(\mathcal{H})` on product Hilbert space :math:`\mathcal{H}=\otimes_i \mathcal{H}_i`
    where :math:`\mathcal{H}_i` is a local Hilbert space of *i*-th degree of freedom. 
    The product operator :math:`O = \otimes_i o_i` is a tensor product of local operators :math:`o_i`.
    Unless explicitly specified, the :math:`o_i` is assumed to be identity operator.

    positions : tuple(int)
        positions of the local operators :math:`o_i` in the product different than identity
    operators : tuple(yast.Tensor)
        local operators in the product, acting at *i*-th positon :code:`positions[i]` 
        different than identity.
    """
    amplitude : float = 1.0
    positions : tuple = ()
    operators : tuple = ()


def generate_H1(I, term):
    r"""
    Apply local operators specified by term in :class:`Hterm` to the mpo I.

    Apply swap_gates to introduce fermionic degrees of freedom
    (fermionic order is the same as order of sites in Mps).
    With this respect, local operators specified in term.operators are applied from last to first,
    i.e., from right to left.

    Parameters
    ----------
    term: :class:`Hterm`
        instruction to create the Mpo which is a product of
        operators element.operator at location element.position
        and with amplitude element.amplitude.
    """
    H1 = I.copy()
    for site, op in zip(term.positions[::-1], term.operators[::-1]):
        op = op.add_leg(axis=0, s=1)
        leg = op.get_legs(axis=0)
        one = ones(config=op.config, legs=(leg, leg.conj()))
        temp = ncon([op, H1[site]], [(-1, -2, 1), (-0, 1, -3, -4)])
        H1[site] = temp.fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
        for n in range(site):
            temp = ncon([H1[n], one], [(-0, -2, -3, -5), (-1, -4)])
            temp = temp.swap_gate(axes=(1, 2))
            H1[n] = temp.fuse_legs(axes=((0, 1), 2, (3, 4), 5), mode='hard')
    for n in H1.sweep():
        H1[n] = H1[n].drop_leg_history(axis=(0, 2))
    H1[0] = term.amplitude * H1[0]
    return H1


def generate_mpo(I, terms, opts=None):
    """
    Generate mpo provided a list of Hterm-s and identity operator I.

    TODO: implement more efficient algorithm
    """
    H1s = [generate_H1(I, term) for term in terms]
    M = add(*H1s)
    M.canonize_sweep(to='last', normalize=False)
    M.truncate_sweep(to='first', opts=opts, normalize=False)
    return M


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
        """
        self.N = N
        self._ops = operators
        self._map = {(i,): i for i in range(N)} if map is None else map
        if len(self._map) != N or sorted(self._map.values()) != list(range(N)):
            raise YampsError("Map is inconsistent with mps of N sites.")
        self._Is = {k: 'I' for k in self._map.keys()} if Is is None else Is
        if self._Is.keys() != self._map.keys():
            raise YampsError("Is is inconsistent with map.")
        if not all(hasattr(self._ops, v) and callable(getattr(self._ops, v)) for v in self._Is.values()):
            raise YampsError("operators do not contain identity specified in Is.")

        self._I = Mpo(self.N)
        for label, site in self._map.items():
            local_I = getattr(self._ops, self._Is[label])
            self._I.A[site] = local_I().add_leg(axis=0, s=1).add_leg(axis=2, s=-1)

        self.config = self._I.A[0].config
        self.parameters = {} if parameters is None else parameters

        self.opts = opts

    def random_seed(self, seed):
        self.config.backend.random_seed(seed)

    def I(self):
        """ return identity Mpo. """
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
        """
        if n is None:
            n = (0,) * self.config.sym.NSYM
        try:
            n = tuple(n)
        except TypeError:
            n = (n,)
        an = np.array(n, dtype=int)

        nl = tuple(an * 0)
        psi = Mps(self.N)

        ll = Leg(self.config, s=1, t=(nl,), D=(1,),)
        for site in psi.sweep(to='last'):
            lp = self._I[site].get_legs(axis=1)
            nr = tuple(an * (site + 1) / self.N)
            if site != psi.last:
                lr = random_leg(self.config, s=-1, n=nr, D_total=D_total, sigma=sigma, legs=[ll, lp])
            else:
                lr = Leg(self.config, s=-1, t=(n,), D=(1,),)
            psi.A[site] = rand(self.config, legs=[ll, lp, lr], dtype=dtype)
            ll = psi.A[site].get_legs(axis=2).conj()
        if sum(ll.D) == 1:
            return psi
        raise YampsError("Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization) ")

    def random_mpo(self, D_total=8, sigma=1, dtype='float64'):
        """
        Generate a random Mpo of virtual bond dimension D_total.

        Mainly, for testing.

        Parameters
        ----------
        D_total : int
            total dimension of virtual space
        sigma : int
            variance of Normal distribution from which dimensions of charge sectors
            are drawn.
        """
        n0 = (0,) * self.config.sym.NSYM
        psi = Mpo(self.N)

        ll = Leg(self.config, s=1, t=(n0,), D=(1,),)
        for site in psi.sweep(to='last'):
            lp = self._I[site].get_legs(axis=1)
            if site != psi.last:
                lr = random_leg(self.config, s=-1, n=n0, D_total=D_total, sigma=sigma, legs=[ll, lp, lp.conj()])
            else:
                lr = Leg(self.config, s=-1, t=(n0,), D=(1,),)
            psi.A[site] = rand(self.config, legs=[ll, lp, lr, lp.conj()], dtype=dtype)
            ll = psi.A[site].get_legs(axis=2).conj()
        if sum(ll.D) == 1:
            return psi
        raise YampsError("Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization).")

    def mps(self, psi_str, parameters=None):
        """
        initialize simple product states 

        TODO: implement
        """
        pass


    def mpo(self, H_str, parameters=None):
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
        self.parameters = parameters
        c2 = latex2single_term(H_str, self)
        c3 = single_term2Hterm(c2, self)
        return generate_mpo(self._I, c3)

def single_term2Hterm(c2, gen):
    fin_list = []
    for ic in c2:
        amplitude, positions, operators = 1, [], []
        for iop in ic.op:
            if len(iop)==1:
                amplitude *= gen.parameters[iop[0]] if iop[0] in gen.parameters else float(iop[0])
            else:
                name, indicies = iop[0], gen._map[iop[1:]]
                positions.append(indicies)
                if name in gen.parameters:
                    operators.append(gen.parameters[name](indicies))
                else:
                    operators.append(gen._ops.to_dict()[name](indicies))
        fin_list.append(Hterm(amplitude, positions, operators))
    return fin_list

def random_dense_mps(N, D, d, **kwargs):
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mps(D_total=D)


def random_dense_mpo(N, D, d, **kwargs):
    G = Generator(N, Qdit(d=d, **kwargs))
    return G.random_mpo(D_total=D)
