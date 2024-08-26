# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Basic structures forming PEPS network. """
from __future__ import annotations
from typing import Sequence, Union
from typing import NamedTuple
from ... import YastnError


class Site(NamedTuple):
    """ Site coordinates `(nx, ny)` are consistent with matrix indexing with `(row, column)`. """
    nx : int = 0
    ny : int = 0

    def __str__(self):
        return f"Site({self.nx}, {self.ny})"


class Bond(NamedTuple):
    """ A bond between two lattice sites. """
    site0 : Site = None
    site1 : Site = None

    def __str__(self):
        return f"Bond(({self.site0[0]}, {self.site0[1]}), ({self.site1[0]}, {self.site1[1]}))"

    def __format__(self, spec):
            return str(self).format(spec)


_periodic_dict = {'infinite': 'ii', 'obc': 'oo', 'cylinder': 'po'}
# 'i' for infinite, 'o' for open, 'p' for periodic; two directions


class SquareLattice():

    def __init__(self, dims=(2, 2), boundary='infinite'):
        r"""
        Geometric information about 2D square lattice.

        Parameters
        ----------
        dims: tuple[int, int]
            Size of the unit cell in a form of ``dims=(rows, columns)``.
            Site(0, 0) corresponds to top-left corner of the unit cell.

        boundary: str
            Finite lattice, infinite lattice, or finite cylinder periodic along rows,
            respectively, for 'obc', 'infinite', or 'cylinder'.
        """
        if boundary not in ('obc', 'infinite', 'cylinder'):
            raise YastnError(f"{boundary=} not recognized; should be 'obc', 'infinite', or 'cylinder'")

        self.boundary = boundary
        self._periodic = _periodic_dict[boundary]
        self._dims = (dims[0], dims[1])
        self._sites = tuple(Site(nx, ny) for ny in range(self._dims[1]) for nx in range(self._dims[0]))
        self._dir = {'tl': (-1, -1), 't': (-1, 0), 'tr': (-1,  1),
                      'l': ( 0, -1),                'r': ( 0,  1),
                     'bl': ( 1, -1), 'b': ( 1, 0), 'br': ( 1,  1)}

        bonds_h, bonds_v = [], []
        for s in self._sites:
            s_r = self.nn_site(s, d='r')  # left is before right in the fermionic order
            if s_r is not None:
                bonds_h.append(Bond(s, s_r))
            s_b = self.nn_site(s, d='b')  # top is before bottom in the fermionic order
            if s_b is not None:
                bonds_v.append(Bond(s, s_b))
        self._bonds_h = tuple(bonds_h)
        self._bonds_v = tuple(bonds_v)

    @property
    def Nx(self) -> int:
        """ Number of rows in the unit cell. """
        return self._dims[0]

    @property
    def Ny(self) -> int:
        """ Number of columns in the unit cell. """
        return self._dims[1]

    @property
    def dims(self) -> tuple[int, int]:
        """ Size of the unit cell as (rows, columns). """
        return self._dims

    def sites(self, reverse=False) -> Sequence[Site]:
        """ Sequence of unique lattice sites. """
        return self._sites[::-1] if reverse else self._sites

    def bonds(self, dirn=None, reverse=False) -> Sequence[Bond]:
        """
        Sequence of unique nearest neighbor bonds between lattice sites.

        Parameters
        ----------
        dirn: None | str
            return horizontal followed by vertical bonds if None;
            'v' and 'h' are for vertical and horizontal bonds only, respectively.

        reverse: bool
            whether to reverse the order of bonds.
        """
        if dirn == 'v':
            return self._bonds_v[::-1] if reverse else self._bonds_v
        if dirn == 'h':
            return self._bonds_h[::-1] if reverse else self._bonds_h
        return self._bonds_v[::-1] + self._bonds_h[::-1] if reverse else self._bonds_h + self._bonds_v

    def nn_site(self, site, d) -> Site | None:
        """
        Index of the lattice site neighboring the :code:`site` in the direction :code:`d`.

        For infinite lattices, this function simply shifts the ``site`` by provided vector ``d``.
        For finite lattices with open/periodic boundary it handles corner cases where ``d`` is too large and the
        resulting Site either doesn't exist or it wraps around periodic boundary.

        Return ``None`` if there is no neighboring site in a given direction.

        Parameters
        ----------
        d: str | tuple[int, int]
            Take values in: 't', 'b', 'l', 'r', 'tl', 'bl', 'tr', 'br',
            or a tuple of shifts (dx, dy).
        """
        if site is None:
            return None
        x, y = site
        dx, dy = self._dir[d] if isinstance(d, str) else d
        x, y = x + dx, y + dy

        if self._periodic[0] == 'o' and (x < 0 or x >= self._dims[0]):
            return None
        if self._periodic[1] == 'o' and (y < 0 or y >= self._dims[1]):
            return None
        if self._periodic[0] == 'p' and (x < 0 or x >= self._dims[0]):
            x = x % self._dims[0]
        # we don't have such option now:
        # if self._periodic[1] == 'p' and (y < 0 or y >= self._dims[1]):
        #     y = y % self._dims[1]
        return Site(x, y)

    def nn_bond_type(self, bond) -> tuple[str, bool]:
        """
        Raise YastnError if a bond does not connect nearest-neighbor lattice sites.
        Return bond orientation in 2D grid as a tuple: dirn, l_ordered
        dirn is 'h' or 'v' (horizontal, vertical).
        l_ordered is True for bond directed as 'lr' or 'tb', and False for 'rl' or 'bt'.
        """
        s0, s1 = bond
        if self.nn_site(s0, 'r') == s1 and self.nn_site(s1, 'l') == s0:
            return 'h', True  # dirn, l_ordered
        if self.nn_site(s0, 'b') == s1 and self.nn_site(s1, 't') == s0:
            return 'v', True
        if self.nn_site(s0, 'l') == s1 and self.nn_site(s1, 'r') == s0:
            return 'h', False
        if self.nn_site(s0, 't') == s1 and self.nn_site(s1, 'b') == s0:
            return 'v', False
        raise YastnError(f"{bond} is not a nearest-neighbor bond.")

    def f_ordered(self, *bond) -> bool:
        """
        Check if bond (s0, s1) appears in fermionic order.
        The convention is consistent with the PEPS diagrams in https://arxiv.org/abs/0912.0646.

        Args:
            bond (Bond or tuple[Site,Site] or Site, Site):
        """
        if len(bond)==1: # (Bond,) or ((Site,Site),) 
            s0, s1= bond[0]
        else: # (Site, Site)
            s0, s1 = bond[0], bond[1]
        return s0[1] < s1[1] or (s0[1] == s1[1] and s0[0] <= s1[0])

    def site2index(self, site):
        """
        Maps any Site of the underlying square lattice (accounting for boundaries)
        into corresponding Site within primitive unit cell.
        """
        if site is None:
            return None
        x = site[0] % self._dims[0] if self._periodic[0] == 'i' else site[0]
        y = site[1] % self._dims[1] if self._periodic[1] == 'i' else site[1]
        return (x, y)

    def __dict__(self):
        """Return a dictionary representation of the object."""
        return {'dims': self.dims, 'boundary': self.boundary, 'sites': self.sites()}


class CheckerboardLattice(SquareLattice):

    def __init__(self):
        r"""
        Geometric information about infinite checkerboard lattice, which
        is an infinite lattice with :math:`2{\times}2` unit cell and two unique tensors.
        """
        super().__init__(dims=(2, 2), boundary='infinite')
        self._sites = (Site(0, 0), Site(0, 1))
        self._bonds_h = (Bond(Site(0, 0), Site(0, 1)), Bond(Site(0, 1), Site(0, 2)))
        self._bonds_v = (Bond(Site(0, 0), Site(1, 0)), Bond(Site(0, 1), Site(1, 1)))

    def site2index(self, site):
        """ Tensor index depending on site. """
        return (site[0] + site[1]) % 2


class RectangularUnitcell(SquareLattice):
    # TODO Optionally numpy.array(dtype=int) ?
    #      Drop integer ? Can be anything, i.e. even string label for tensors

    def __init__(self, pattern, boundary='infinite'):
        r"""
        Rectangular unit cells supporting patterns characterized by a single momentum ``Q=(q_x,q_y)``.

        Inspired by https://github.com/b1592/ad-peps by B. Ponsioen.

        Parameters
        ----------
        pattern: Sequence[Sequence[int]] | dict[tuple[int,int],int]
            Definition of a rectangular unit cell that tiles the square lattice.
            Integers are labels of unique tensors populating the sites within the unit cell.

            Examples of such patterns can be:

                * [[0,],] : 1x1 unit cell, Q=0
                * [[0,1],] : 1x2 unit cell, Q=(\pi, 0)
                * [[0,1],[1,0]] : 2x2 unit cell with bipartite pattern, Q=(\pi, \pi)
                * [[0,1,2],[1,2,0],[2,0,1]] : 3x3 unit cell with diagonal stripe order, Q=(2\pi/3, 2\pi/3)

        Warning
        -------
        It is assumed that the neighborhood of each unique tensor is identical.
        This excludes cases as ``[[0, 1], [1, 1]]``.
        """
        # TODO validation
        #   pattern should be len > 0, all rows should be of the same length
        #   the set of integers in pattern should be equal to set of keys of tensors
        if isinstance(pattern, dict):
            # validate ranges
            min_row, min_col = map(min, zip(*pattern.keys()))
            max_row, max_col = map(max, zip(*pattern.keys()))
            assert min_row == 0 and min_col == 0, "Invalid pattern specification"
            pattern = [[pattern[(r, c)] for c in range(max_col + 1)] for r in range(max_row + 1)]
        super().__init__(dims=(len(pattern), len(pattern[0])), boundary='infinite')
        #
        self._site2index = {(row, col): t for row, row_elems in enumerate(pattern) for col, t in enumerate(row_elems)}
        #
        # unique sites
        tmp = {b: a for a, b in sorted(self._site2index.items(), reverse=True)}
        self._sites = tuple(sorted(tmp.values()))
        #
        # unique bonds
        self._bonds_h = tuple(Bond( s,self.nn_site(s, 'r')) for s in self._sites)
        self._bonds_v = tuple(Bond( s,self.nn_site(s, 'b')) for s in self._sites)

    def site2index(self, site) -> int:
        """ Tensor index depending on site. """
        return self._site2index[site[0] % self.Nx, site[1] % self.Ny]

    def __str__(self):
        return type(self).__name__ + f"(Nx={self.Nx}, Ny={self.Ny})\n"\
            +f"unique sites={self._sites})\n"\
            +f"unique horizontal bonds={self._bonds_h}\n"\
            +f"unique vertical bonds={self._bonds_v}\n"\
            + "\n".join([ ", ".join([f"{self.site2index((row,col))}" for col in range(self.Ny) ]) for row in range(self.Nx)])

    def __repr__(self):
        return f"RectangularUnitcell(pattern={self._site2index})"

    def __dict__(self):
        """Return a dictionary representation of the object."""
        return {'pattern': self._site2index, 'boundary': self.boundary }
