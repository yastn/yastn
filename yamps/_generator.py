import numpy as np
from typing import NamedTuple
from ._mps import Mpo
from ._auxliary import add
from .basis import use_default_basis
import yast


class mpo_term(NamedTuple):
    r"""
    The length of 'position' should be the same as 'operator'
    """
    amplitude: float = 1.0
    r"""
    A number multiplier (float or complex) for the MPO term.
    """
    position: tuple = (0,)
    r"""
    tuple of int-s. Difines the positition for each operator defined in 'operator'. Indexing of the MPO sites starts from 0. 
    """
    operator: tuple = ('identity',)
    r""""
    tuple of string-s. Gives definition for each operator wchich should exist in fullbasis for the generator you use. 
    """



class GenerateOpEnv():
    """
    Class provides an environment for building MPO (yamps.Mpo) based on the configuration for Tensor's and with operator basis provided for the environment.
    """

    def __init__(self, N, config, opts={'tol': 1e-14}):
        r"""
        Parameters
        ----------
        N: int
            length of Mpo

        config: NamedTuple
            contains information for the configuration of :meth:`yast.Tensor`

        opts: dict
            options passed to :meth:`yast.linalg.svd` to truncate virtual bond dimensions when compressing Mpo

        """
        self.N = N
        self.opts = opts
        self.config = config
        self.fullbasis = {}

    def identity_mpo(self):
        # prepare identity
        self.identity = Mpo(self.N)
        for n in range(self.N):
            id = self.fullbasis['identity'](n).copy().add_leg(axis=0, s=1).add_leg(axis=-1, s=-1)
            self.identity.A[n] = id

    def use_default(self, basis_type='creation_annihilation'):
        self.fullbasis = use_default_basis(self.config, basis_type)
        self.identity_mpo()
    
    def use_basis(self, basis):
        r"""
        Use the single-particle operators defines by user. Definitions are supplier to self.fullbasis

        Parameters
        ----------
        basis: dict
            The list of operators defined for the generator. The keys should be strings while entries shuold be either:

            1/ :meth:`yast.Tensor` with two phisical legs only,

            2/ a list or tuple of length self.N containing :meth:`yast.Tensor` for each element of Mpo lattice,

            3/ a function with the index as an argument assigning a :meth:`yast.Tensor` for each index value. 

            You always have to define 'identity' tensor for your generator. Alternatively you can use predefines basis given by self.use_default()
        """
        if 'identity' not in basis:
            raise YampsError("Basis doesn't contain identity yast.Tensor. Provide an entry with key 'identity'.")
        # prepare full basis for the envoronment
        for key, content in basis.keys():
            if isinstance(content, yast.Tensor):
                self.fullbasis[key] = lambda j: content
            elif isinstance(content, list) or isinstance(content, tuple):
                if len(content) != self.N:
                    raise YampsError("Number or operators doesn't match Mpo Hilbert space. If you provide basis in a form of a list make sure that you define it for all local subspace of Mpo.")
                self.fullbasis[key] = lambda j: content[j]
            else:
                # I am not checking if it is a function. If so I would have to import types and do isinstance(f, types.FunctionType)
                # do I have to check if a function has only one argument?
                self.fullbasis[key] = content
        self.identity_mpo()

    def generate(self, element):
        r"""
        Parameters
        ----------
        element: <class 'mpo_term'>
            instruction to create the Mpo which is a product of operators element.operator at location element.position and with amplitude element.amplitude.
        """
        product = self.identity.copy()
        for j, op in list(zip(element.position, element.operator))[-1::-1]:
            operator = self.fullbasis[op](j).copy().add_leg(axis=0, s=1)
            leg = operator.get_legs(0)
            cfg = operator.config
            product.A[j] = yast.ncon([product.A[j], operator], [(-1, 1, -4, -5), (-2, -3, 1)])
            product.A[j] = product.A[j].fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
            for j3 in range(j):
                virtual = yast.ones(config=cfg, legs=[leg, leg.conj()])
                product.A[j-1-j3] = yast.ncon([product.A[j-1-j3], virtual], [(-1, -3, -4, -5), (-2, -6)])
                product.A[j-1-j3] = product.A[j-1-j3].swap_gate(axes=(1, 2))
                product.A[j-1-j3] = product.A[j-1-j3].fuse_legs(axes=((0, 1), 2, 3, (4, 5)), mode='hard')
        for n in range(self.N):
            product.A[n] = product.A[n].drop_leg_history(axis=(0,3))
        return element.amplitude * product

    def sum(self, sum_elements):
        res = []
        for element in sum_elements:
            res.append(self.generate(element))
        M = add(*res)
        M.canonize_sweep(to='last', normalize=False)
        M.truncate_sweep(to='first', opts=self.opts, normalize=False)
        return M

    def latex2yamps(self, H_str, parameters={}):
        r"""
        Convert latex-like string to yamps MPO. Use the single-particle operators defines by user. Definitions are supplier to self.fullbasis

        Parameters
        -----------
        H_str: str
            The definition of the MPO given as latex expression. The definition uses string names of the operators given in self.fullbasis. The assigment of the location is 
            given e.g. for 'cp' operator as 'cp_j' for 'cp' operator acring on site 'j'. A product of operators will be written as 'cp_j.c_(j+1)' where operators are separated by a dot. 
            To mulply by a number use 'g * cp_j.c_{j+1}' where 'g' can be defines in 'parameters'. You can write all elements explicitely separating by '+' or use use '\sum_{j=0}^5' to sum from index 0 to 5 (included). 
            e.g. \sum_{j=0}^5 g * cp_{j}.c_{j+1} '.
        parameters: dict
            Keys for the dict define the extressions occuring in H_str
        
        Returns
        --------
            yamps.Mpo
        """
        # remove excess spaces
        while "  " in H_str:
            H_str = H_str.replace("  ", " ")
        H_str = H_str.split(" + ")
        mpo_term_list = []
        for h in H_str:
            # search for amplitude
            h = h.replace("*", "* ")
            h = h.split(" ")
            amp = [i for i in h if "*" in i]
            [h.remove(ia) for ia in amp]  # remove terms of amplitude we don't need them any more
            amp = ' '.join(amp).replace("*","").split(' ')
            if '' in amp:
                amp.remove('') # remove empty elements
            amplitude = np.prod(np.array([1.0]+[parameters[ia] for ia in amp]))
            if '\sum' in h[0]:
                sum_name = h[0].replace('{', '').replace('}', '')
                sum_name = sum_name.replace("_", " _")
                sum_name = sum_name.replace("^", " ^")
                sum_name = sum_name.split(' ')
                lower_bound = [i for i in sum_name if "_" in i][0].replace('_', '')
                iterator, lower_bound = lower_bound.split('=')
                upper_bound = [i for i in sum_name if "^" in i][0].replace('^', '')
                lower_bound, upper_bound = int(lower_bound), int(upper_bound)
                op_list = []
                for it in range(lower_bound, upper_bound+1):
                    op_list.append( h[1].replace(iterator, str(it)) )
            else:
                op_list = h
            for iop_list in op_list:    
                h_op=iop_list.split('.')
                h_op = [ih_op.split('_') for ih_op in h_op]
                position = tuple([eval(ih_op[1].replace('{','').replace('}', '')) for ih_op in h_op])
                operator = tuple([ih_op[0] for ih_op in h_op])
                mpo_term_list.append(mpo_term(amplitude, position, operator))
        return self.sum(mpo_term_list)
