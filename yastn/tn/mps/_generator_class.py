from __future__ import annotations
from ... import YastnError
from ._mps import Mpo
from ._latex2term import latex2term
from ._initialize import random_mpo, random_mps
from ._generate_mpo import Hterm, generate_mpo


class Generator:

    def __init__(self, N, operators, map=None, Is=None, parameters=None, opts={"tol": 1e-13}):
        r"""
        Generator is a convenience class building MPOs from a set of local operators.

        Parameters
        ----------
        N : int
            number of sites of MPO.
        operators : object or dict[str, yastn.Tensor]
            a set of local operators, e.g., an instance of :class:`yastn.operators.Spin12`.
            Or a dictionary with string-labeled local operators, including at least ``{'I': <identity operator>,...}``.
        map : dict[int, int]
            custom labels of N sites indexed from 0 to N-1 , e.g., ``{3: 0, 21: 1, ...}``.
            If ``None``, the sites are labled as :code:`{site: site for site in range(N)}`.
        Is : dict[int, str]
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

    def I(self) -> yastn.tn.mps.MpsMpo:
        """ Indetity MPO derived from identity in local operators class. """
        return self._I.copy()

    def random_mps(self, n=None, D_total=8, sigma=1, dtype='float64') -> yastn.tn.mps.MpsMpo:
        r"""
        Generate a random MPS of total charge ``n`` and bond dimension ``D_total``.

        See, :meth:`mps.random_mps<yastn.tn.mps.random_mps>`.
        """
        return random_mps(self._I, n=n, D_total=D_total, sigma=sigma, dtype=dtype)

    def random_mpo(self, D_total=8, sigma=1, dtype='float64') -> yastn.tn.mps.MpsMpo:
        r"""
        Generate a random MPO with bond dimension ``D_total``.

        See, :meth:`mps.random_mps<yastn.tn.mps.random_mpo>`.
        """
        return random_mpo(self._I, D_total=D_total, sigma=sigma, dtype=dtype)

    def mpo_from_latex(self, H_str, parameters=None, opts=None) -> yastn.tn.mps.MpsMpo:
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

