import numpy as np
from typing import NamedTuple
from ... import initialize, operators, tensor, YastError
from ._mps import Mpo, Mps, add

from itertools import compress, product
from re import match


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
        one = initialize.ones(config=op.config, legs=(leg, leg.conj()))
        temp = tensor.ncon([op, H1[site]], [(-1, -2, 1), (-0, 1, -3, -4)])
        H1[site] = temp.fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
        for n in range(site):
            temp = tensor.ncon([H1[n], one], [(-0, -2, -3, -5), (-1, -4)])
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
        self._map = {i:i for i in range(N)} if map is None else map
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

        ll = tensor.Leg(self.config, s=1, t=(nl,), D=(1,),)
        for site in psi.sweep(to='last'):
            lp = self._I[site].get_legs(axis=1)
            nr = tuple(an * (site + 1) / self.N)
            if site != psi.last:
                lr = tensor.random_leg(self.config, s=-1, n=nr, D_total=D_total, sigma=sigma, legs=[ll, lp])
            else:
                lr = tensor.Leg(self.config, s=-1, t=(n,), D=(1,),)
            psi.A[site] = initialize.rand(self.config, legs=[ll, lp, lr], dtype=dtype)
            ll = psi.A[site].get_legs(axis=2).conj()
        if sum(ll.D) == 1:
            return psi
        raise YastError("MPS: Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization) ")

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

        ll = tensor.Leg(self.config, s=1, t=(n0,), D=(1,),)
        for site in psi.sweep(to='last'):
            lp = self._I[site].get_legs(axis=1)
            if site != psi.last:
                lr = tensor.random_leg(self.config, s=-1, n=n0, D_total=D_total, sigma=sigma, legs=[ll, lp, lp.conj()])
            else:
                lr = tensor.Leg(self.config, s=-1, t=(n0,), D=(1,),)
            psi.A[site] = initialize.rand(self.config, legs=[ll, lp, lr, lp.conj()], dtype=dtype)
            ll = psi.A[site].get_legs(axis=2).conj()
        if sum(ll.D) == 1:
            return psi
        raise YastError("MPS: Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization).")


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
        # get everything to dictionary
        parameters = self.parameters if parameters is None else {**self.parameters, **parameters}
        params_dict = parameters
        basis_dict = self._ops.to_dict()
        # merge to a single dictionary
        info_dict = {**params_dict.copy(), **basis_dict.copy()}

        # interpret string
        # RULES:
        # * put all parameters in info_dict are functions, if constant then should be single-parameter function
        tmp = H_str
        tmp = tmp.replace("*", " ")
        for ic in ["*", "(", ")", "\in"]:  # put additional spaces
            tmp = tmp.replace(ic, " "+ic+" ")
        while "  " in tmp:  # remove double spaces
            tmp = tmp.replace("  ", " ")
        tmp = tmp.replace(" \in ", ".in.").replace("\sum_", ".sum_.") # remove \ because I don't know how to manege this
        lall = tmp.strip().split(" ")  # remove space if it is on the beginning/end & finally split the instruction

        # split sums away from each other
        sep, H = {}, None
        terminate, hodor = ['+', '-'], False
        isep, id_sep = 0, 0  # where new id_step-th substring begins
        # split to separate sums
        sep[id_sep] = []
        for it in range(len(lall)):
            tmp = lall[it]
            if tmp == '(':
                hodor, terminate = True, [')']
            if hodor and tmp in terminate:
                hodor, terminate = False, ['+', '-']
            if not hodor and tmp in terminate and it > isep:
                isep = it+1
                id_sep += 1
                sep[id_sep] = []
            sep[id_sep].append(tmp)
        # analyze terms, get info on sums
        for k in sep.keys():
            spart = sep[k]
            issum = ['sum' in ix for ix in spart]
            is_sum = list(compress(spart, issum))
            not_sum = list(compress(spart, [not ix for ix in issum]))
            # sum instruction
            is_sum = [match(r".sum_.{(?P<index>.*).in.(?P<range>.*)}", tmp).group("index", "range") for tmp in is_sum]
            is_index = [tmp[0] for tmp in is_sum]
            is_range = [params_dict[tmp[1]] for tmp in is_sum]
            read_not_sum = [None]*len(not_sum)
            sgn_math = ['+', '-', '(', ')']
            sgn_info = list(info_dict.keys())
            # ger order of operations
            iorder = 0
            for inum, ival in enumerate(not_sum[::-1]):
                if ival in sgn_math:
                    iorder += 1
                read_not_sum[-inum-1] = iorder
                if ival in sgn_math:
                    iorder += 1
            # remove bracket for convenience
            remove_braket = [ix not in ['(', ')'] for ix in not_sum]
            not_sum = list(compress(not_sum, remove_braket))
            read_not_sum = list(compress(read_not_sum, remove_braket))
            # iterate over is_index-es each over is_range-es range of numbers
            range_span = product(*is_range, repeat=1)
            for idx in range_span:
                # it does a seach for a parametrs and operators for each range-span
                # I am not sure how to change that because each elemetn is a function...
                for iorder in np.unique(read_not_sum):
                    tmp = [ira == iorder for ira in read_not_sum]
                    itake = list(compress(not_sum, tmp))
                    op_holder, ind_holder, amp_holder = [], [], 1
                    iind = (0,)  # put dummy just in case parameter is just a number
                    for inum, ival in enumerate(itake[::-1]):
                        # extract using map if needed
                        get_info = match(r"(?P<ival>.*)_{(?P<iind>.*)}", ival)
                        if get_info:
                            ival, iind = get_info.group("ival"), get_info.group("iind")
                            iind = iind.split(',')
                            if all([i.isnumeric() for i in iind]):
                                # convert to numerical index
                                iind = [int(i) for i in iind]
                            else:
                                # find index from sum indexing
                                for i_ind in range(len(iind)):
                                    for i_index in range(len(is_index)):
                                        if is_index[i_index] in iind[i_ind]:
                                            iind[i_ind] = iind[i_ind].replace(is_index[i_index], str(idx[i_index]))
                                    iind[i_ind] = eval(iind[i_ind])
                        if ival in sgn_info:
                            ival = info_dict[ival](*iind)
                            if type(ival) == tensor.Tensor:
                                op_holder.append(ival)
                                ind_holder.append(*iind)
                            else:
                                amp_holder *= ival
                        else:
                            if '+' == ival:
                                amp_holder *= float(1)
                            elif '-' == ival:
                                amp_holder *= float(-1)
                            else:
                                amp_holder *= float(ival)
                    # generate mpo product using automatic
                    ind_holder, op_holder = ind_holder[::-1], op_holder[::-1]
                    place_holder = [Hterm(amp_holder, ind_holder, op_holder)]
                    if not op_holder:
                        H = amp_holder * H
                    else:
                        if not H:
                            H = generate_mpo(self._I, place_holder, self.opts)
                        elif H and amp_holder > 0.0:
                            H = H + generate_mpo(self._I, place_holder, self.opts)
            H.canonize_sweep(to='last', normalize=False)
            H.truncate_sweep(to='first', opts=self.opts, normalize=False)
        return H

    def mps(self, psi_str, parameters=None):
        """
        initialize simple product states 

        TODO: implement
        """
        pass


def random_dense_mps(N, D, d, **kwargs):
    G = Generator(N, operators.Qdit(d=d, **kwargs))
    return G.random_mps(D_total=D)


def random_dense_mpo(N, D, d, **kwargs):
    G = Generator(N, operators.Qdit(d=d, **kwargs))
    return G.random_mpo(D_total=D)
