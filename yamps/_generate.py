import numpy as np
from typing import NamedTuple
import yast
from ._mps import Mpo, Mps, YampsError
from ._auxliary import add


class Hterm(NamedTuple):
    amplitude : float = 1.0
    positions : tuple = ()
    operators : tuple = ()


def generate_H1(I, term):
    r"""
    Apply local operators specified by term to the mpo I (makes a copy).

    Apply swap gates

    Parameters
    ----------
    term: :class:`Hterm`
        instruction to create the Mpo which is a product of operators element.operator at location element.position and with amplitude element.amplitude.
    """
    H1 = I.copy()
    for site, op in zip(term.positions[::-1], term.operators[::-1]):
        op = op.add_leg(axis=0, s=1)
        leg = op.get_legs(axis=0)
        one = yast.ones(config=op.config, legs=(leg, leg.conj()))
        temp = yast.ncon([op, H1[site]], [(-2, -3, 1), (-1, 1, -4, -5)])
        H1[site] = temp.fuse_legs(axes=((0, 1), 2, 3, 4), mode='hard')
        for n in range(site):
            temp = yast.ncon([H1[n], one], [(-1, -3, -4, -5), (-2, -6)])
            temp = temp.swap_gate(axes=(1, 2))
            H1[n] = temp.fuse_legs(axes=((0, 1), 2, 3, (4, 5)), mode='hard')
    for n in H1.sweep():
        H1[n] = H1[n].drop_leg_history(axis=(0, 3))
    H1[0] = term.amplitude * H1[0]
    return H1


def generate_mpo(I, terms, opts):
    """
    Generate mpo provided a list of Hterm-s.
    """
    H1s = [generate_H1(I, term) for term in terms]
    M = add(*H1s)
    M.canonize_sweep(to='last', normalize=False)
    M.truncate_sweep(to='first', opts=opts, normalize=False)
    return M


class Generator:
    """ Generator to create Mpo-s and Mps-s from local operators. """
    def __init__(self, N, operators, map=None, Is=None, opts={"tol": 1e-14}):
        """
        N : int
            number of sites of mps/mpo
        operators : class
            generator of local operators, such as an instance of yast.operators.Spin12
        map : dict
            custom labels for mps sites, {site_label: mps_site}, where mps_site are ordered as 0, 1, ..., N - 1
            If None, use default identity map, with labels 0, 1, ..., N - 1
        Is : dict
            For each mps site, name identity operator in operators class, {site_label: str}.
            If local mps sites have different physical dimensions, each should have separate identity operator defined in operators.
            If None, use default {site_label: 'I'}
        opts : dict
            used if compression is needed. Options passed to :meth:`yast.linalg.svd`
        """
        self.N = N
        self._ops = operators
        self._map = {i:i for i in range(N)} if map is None else map
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
            self._I.A[site] = local_I().add_leg(axis=0, s=1).add_leg(axis=-1, s=-1)

        self.config = self._I.A[0].config
        self.opts=opts

    def I(self):
        """ return identity Mpo. """
        return self._I.copy()

    def random_mps(self, n=None, D_total=8, sigma=1, dtype='float64'):
        """
        Generate a random Mps of total charge n and virtual bond dimension D_total.
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

        ll = yast.Leg(self.config, s=1, t=(nl,), D=(1,),)
        for site in psi.sweep(to='last'):
            lp = self._I[site].get_legs(axis=self._I.phys[0])
            nr = tuple(an * (site + 1) / self.N)
            if site != psi.last:
                lr = yast.random_leg(self.config, s=-1, n=nr, D_total=D_total, sigma=sigma, legs=[ll, lp])
            else:
                lr = yast.Leg(self.config, s=-1, t=(n,), D=(1,),)
            psi.A[site] = yast.rand(self.config, legs=[ll, lp, lr], dtype=dtype)
            ll = psi.A[site].get_legs(axis=psi.right[0]).conj()
        if sum(ll.D) == 1:
            return psi
        raise YampsError("Random mps is a zero state. Check parameters (or try running again in this is due to randomness of the initialization) ")

    def mpo(self, H_str, parameters={}):
        r"""
        Convert latex-like string to yamps MPO.

        Parameters
        -----------
        H_str: str
            The definition of the MPO given as latex expression. The definition uses string names of the operators given in. The assignment of the location is 
            given e.g. for 'cp' operator as 'cp_j' for 'cp' operator acting on site 'j'. A product of operators will be written as 'cp_j.c_(j+1)' where operators are separated by a dot. 
            To multiply by a number use 'g * cp_j.c_{j+1}' where 'g' can be defines in 'parameters'. You can write all elements explicitly separating by '+' or use use '\sum_{j=0}^5' to sum from index 0 to 5 (included). 
            e.g. \sum_{j=0}^5 g * cp_{j}.c_{j+1} '.
        parameters: dict
            Keys for the dict define the expressions that occur in H_str

        Returns
        --------
            :class:`yamps.Mpo`
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
                    op_list.append(h[1].replace(iterator, str(it)))
            else:
                op_list = h
            for iop_list in op_list:
                h_op=iop_list.split('.')
                h_op = [ih_op.split('_') for ih_op in h_op]
                positions = tuple(eval(ih_op[1].replace('{','').replace('}', '')) for ih_op in h_op)
                operators = tuple(getattr(self._ops, ih_op[0])() for ih_op in h_op)
                mpo_term_list.append(Hterm(amplitude, positions, operators))
        return generate_mpo(self._I, mpo_term_list, self.opts)
