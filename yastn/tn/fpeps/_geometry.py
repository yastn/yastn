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
from typing import NamedTuple, Sequence
from warnings import warn

from .envs._env_dataclasses import DATA_CLASSES
from ...tensor import YastnError


class Site(NamedTuple):
    """
    Site coordinates `(nx, ny)` are consistent with matrix indexing with `(row, column)`::

        ┌───── y(cols) ──────ᐳ
        │
        x(rows)  (0,0) (0,1) ...
        │        (1,0) (1,1) ...
        │         ...
        ᐯ

    """
    nx : int = 0
    ny : int = 0

    def __str__(self):
        return f"Site({self.nx}, {self.ny})"

def is_site(site):
    return isinstance(site, tuple) and len(site) == 2 and all(isinstance(nn, int) for nn in site)


class Bond(NamedTuple):
    """ A bond between two lattice sites. """
    site0 : Site = None
    site1 : Site = None

    def __str__(self):
        return f"Bond(({self.site0[0]}, {self.site0[1]}), ({self.site1[0]}, {self.site1[1]}))"

    def __format__(self, spec):
            return str(self).format(spec)

def is_bond(bond):
    return isinstance(bond, (tuple, list)) and len(bond) == 2 and all(is_site(site) for site in bond)


_periodic_dict = {'infinite': 'ii', 'obc': 'oo', 'cylinder': 'po'}
# 'i' for infinite, 'o' for open, 'p' for periodic; two directions


class SquareLattice():

    def __init__(self, dims=(2, 2), boundary='infinite', **kwargs):
        r"""
        Geometric information about 2D square lattice.

        Parameters
        ----------
        dims: tuple[int, int]
            Size of the unit cell in a form of ``dims=(rows, columns)``. Site(0, 0) corresponds to top-left corner of the unit cell.

        boundary: str
            Type of boundary conditions:
                * 'infinite' (the default) for an infinite lattice,
                * 'obc' for a finite lattice, or
                * 'cylinder' for a finite cylinder periodic along rows, i.e.::

                    ┌───── y(cols) ──────ᐳ
                    │
                    │        (0, 0) (0, 1) ... (0, Ny-1)
                    x(rows)  (1, 0) (1, 1) ... (1, Ny-1)
                    │         ...
                    │        (Nx-1, 0)   ...   (Nx-1, Ny-1)
                    ᐯ        (0, 0)      ...   (0, Ny-1)

        """
        if boundary not in ('obc', 'infinite', 'cylinder'):
            raise YastnError(f"{boundary=} not recognized; should be 'obc', 'infinite', or 'cylinder'")

        self.boundary = boundary
        self._periodic = _periodic_dict[boundary]
        self._dims = (dims[0], dims[1])
        self._sites = tuple(Site(nx, ny) for ny in range(self.Ny) for nx in range(self.Nx))
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

    def __eq__(self, other):
        return isinstance(other, SquareLattice) and \
               type(self) == type(other) and \
               self._periodic == other._periodic and \
               self._dims == other._dims and \
               self._sites == other._sites and \
               all(self.site2index((nx, ny)) == other.site2index((nx, ny))
                   for ny in range(self.Ny) for nx in range(self.Nx))

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
            'v' and 'h' are, respectively, for vertical and horizontal bonds only.

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
        Index of the lattice site neighboring the ``site`` in the direction ``d``.

        For infinite lattices, this function simply shifts the ``site`` by provided vector ``d``.
        For finite lattices with open/periodic boundary it handles corner cases where ``d`` is too large and the
        resulting site either doesn't exist or it wraps around periodic boundary.

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

    def nn_bond_dirn(self, s0, s1=None) -> str:
        """
        Raise YastnError if a bond does not connect nearest-neighbor lattice sites.
        Return bond orientation in 2D grid as a tuple: dirn, l_ordered
        dirn is 'h' or 'v' (horizontal, vertical).
        l_ordered is True for bond directed as 'lr' or 'tb', and False for 'rl' or 'bt'.
        """
        if s1 is None:
            s0, s1 = s0   # allow syntax where s0 is a bond s0 = (s0, s1)
        if self.nn_site(s0, 'r') == s1 and self.nn_site(s1, 'l') == s0:
            return 'lr'  # dirn
        if self.nn_site(s0, 'b') == s1 and self.nn_site(s1, 't') == s0:
            return 'tb'
        if self.nn_site(s0, 'l') == s1 and self.nn_site(s1, 'r') == s0:
            return 'rl'
        if self.nn_site(s0, 't') == s1 and self.nn_site(s1, 'b') == s0:
            return 'bt'
        raise YastnError(f"{s0}, {s1} are not nearest-neighbor sites.")

    def f_ordered(self, s0, s1) -> bool:
        """
        Check if sites s0, s1 are fermionically ordered (or identical).
        """
        return s0[1] < s1[1] or (s0[1] == s1[1] and s0[0] <= s1[0])

    def site2index(self, site):
        """
        Maps any Site of the underlying square lattice (accounting for boundaries)
        into corresponding Site within primitive unit cell.
        """
        if site is None:
            return None
        x = site[0] % self._dims[0] if self._periodic[0] in 'ip' else site[0]
        y = site[1] % self._dims[1] if self._periodic[1] == 'i' else site[1]
        return (x, y)

    def to_dict(self):
        """ Return a dictionary representation of the object. """
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'dims': self.dims,
                'boundary': self.boundary}


class CheckerboardLattice(SquareLattice):

    def __init__(self, **kwargs):
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

    def to_dict(self):
        """ Return a dictionary representation of the object. """
        return {'type': type(self).__name__,
                'dict_ver': 1}


class RectangularUnitcell(SquareLattice):

    def __init__(self, pattern, **kwargs):
        r"""
        Rectangular unit cells supporting patterns characterized by a single momentum ``Q=(q_x, q_y)``.

        Inspired by https://github.com/b1592/ad-peps by B. Ponsioen.

        Parameters
        ----------
        pattern: Sequence[Sequence[int]] | dict[tuple[int, int], int]
            Definition of a rectangular unit cell that tiles the square lattice.
            Integers are labels of unique tensors populating the sites within the unit cell.

            Examples of such patterns can be:

                * [[0,],] : 1x1 unit cell, Q=0
                * {(0, 0): 0} : 1x1 unit cell, Q=0
                * [[0, 1],] : 1x2 unit cell, Q=(0, \pi)
                * {(0, 0): 0, (0, 1): 1} : 1x2 unit cell, Q=(0, \pi)
                * [[0, 1], [1, 0]] : 2x2 unit cell with bipartite pattern, Q=(\pi, \pi). Equivalent to :class:`yastn.tn.fpeps.CheckerboardLattice`.
                * [[0, 1, 2], [1, 2, 0], [2, 0, 1]] : 3x3 unit cell with diagonal stripe order, Q=(2\pi/3, 2\pi/3)


        Warning
        -------
        It is assumed that the neighborhood of each unique tensor is identical.
        This excludes cases such as ``[[0, 1], [1, 1]]``.
        """
        if isinstance(pattern, dict):
            min_row, min_col = map(min, zip(*pattern.keys()))
            max_row, max_col = map(max, zip(*pattern.keys()))
            if (min_row, min_col) != (0, 0):
                raise YastnError("RectangularUnitcell: pattern keys should cover a rectangle index (0, 0) to (Nx - 1, Ny - 1).")
            try:
                pattern = [[pattern[(r, c)] for c in range(max_col + 1)] for r in range(max_row + 1)]
            except KeyError:
                raise YastnError("RectangularUnitcell: pattern keys should cover a rectangle index (0, 0) to (Nx - 1, Ny - 1).")

        try:
            Nx = len(pattern)
            Ny = len(pattern[0])
        except TypeError:
            raise YastnError("RectangularUnitcell: pattern should form a two-dimensional square matrix of labels.")
        if any(len(row) != Ny for row in pattern):
            raise YastnError("RectangularUnitcell: pattern should form a two-dimensional square matrix of labels.")

        super().__init__(dims=(Nx, Ny), boundary='infinite')
        #
        self._site2index = {(nx, ny): label for nx, row in enumerate(pattern) for ny, label in enumerate(row)}
        #
        try:
            label_sites, label_envs = {}, {}
            for nx in range(Nx):
                for ny in range(Ny):
                    label = self._site2index[nx, ny]
                    env = (self.site2index((nx - 1, ny)), self.site2index((nx, ny - 1)), self.site2index((nx + 1, ny)), self.site2index((nx, ny + 1)))
                    if label in label_sites:
                        label_sites[label].append(Site(nx, ny))
                        label_envs[label].append(env)
                    else:
                        label_sites[label] = [Site(nx, ny)]
                        label_envs[label] = [env]
        except TypeError:
            raise YastnError("RectangularUnitcell: pattern labels should be hashable.")
        if any(len(set(envs)) > 1 for envs in label_envs.values()):
            raise YastnError("RectangularUnitcell: each unique label should have the same neighbors.")
        #
        # unique sites
        self._sites = tuple(sorted(min(sites) for sites in label_sites.values()))
        #
        # unique bonds
        self._bonds_h = tuple(Bond(s, self.nn_site(s, 'r')) for s in self._sites)
        self._bonds_v = tuple(Bond(s, self.nn_site(s, 'b')) for s in self._sites)

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

    def to_dict(self):
        """ Return a dictionary representation of the object. """
        # For serialiation to JSON, dict keys must be str/int/...
        # Hence, we store pattern in format Sequence[Sequence[int]].
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'pattern': [[self.site2index((row, col)) for col in range(self.Ny)] for row in range(self.Nx)]}


class TriangularLattice(SquareLattice):

    def __init__(self, dims=(3, 3), boundary='infinite', full_patch=False, **kwargs):
        r"""
        Geometric information about infinite triangular lattice, which
        is an infinite lattice with :math:`3{\times}3` (full_patch) unit cell and nine unique tensors,
        or :math:`\sqrt{3}{\times}\sqrt{3}` unit cell and three unique tensors;
        a finite lattice with :math:`Nx{\times}Ny` patch and :math:`Nx{\times}Ny` unique tensors.
        """
        self.full_patch = full_patch
        super().__init__(dims=dims, boundary=boundary)
        if self.full_patch:
            bonds_d = []
            for s in self._sites:
                s_r = self.nn_site(s, d='r')  # left is before right in the fermionic order
                s_b = self.nn_site(s, d='b')  # top is before bottom in the fermionic order
                bonds_d.append(Bond(s_b, s_r))
            self._bonds_d = bonds_d
        else:
            self._sites = (Site(0, 0), Site(0, 1), Site(0, 2))
            self._bonds_h = (Bond(Site(0, 0), Site(0, 1)), Bond(Site(0, 1), Site(0, 2)), Bond(Site(0, 2), Site(0, 3)))
            self._bonds_v = (Bond(Site(0, 0), Site(1, 0)), Bond(Site(0, 1), Site(1, 1)), Bond(Site(0, 2), Site(1, 2)))
            self._bonds_d = (Bond(Site(1, 0), Site(0, 1)), Bond(Site(1, 1), Site(0, 2)), Bond(Site(1, 2), Site(0, 3)))

    def site2index(self, site):
        """ Tensor index depending on site. """
        if self.full_patch:
            return (site[0] % self.Nx) * self.Ny + site[1] % self.Ny
        else:
            return (site[1] - site[0]) % 3

    def bonds(self, dirn=None, reverse=False) -> Sequence[Bond]:
        """
        Sequence of unique nearest neighbor bonds between lattice sites.

        Parameters
        ----------
        dirn: None | str
            return horizontal followed by vertical and diagonal bonds if None;
            'v', 'h' and 'd' are, respectively, for vertical, horizontal and diagonal bonds only.

        reverse: bool
            whether to reverse the order of bonds.
        """
        if dirn == 'd':
            return self._bonds_d[::-1] if reverse else self._bonds_d
        if dirn == 'v':
            return self._bonds_v[::-1] if reverse else self._bonds_v
        if dirn == 'h':
            return self._bonds_h[::-1] if reverse else self._bonds_h
        return self._bonds_d[::-1] + self._bonds_v[::-1] + self._bonds_h[::-1] if reverse else self._bonds_h + self._bonds_v + self._bonds_d

    def to_dict(self):
        """ Return a dictionary representation of the object. """
        # For serialiation to JSON, dict keys must be str/int/...
        # Hence, we store pattern in format Sequence[Sequence[int]].
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'pattern': [[self.site2index((row, col)) for col in range(self.Ny)] for row in range(self.Nx)]}


LATTICE_CLASSES = {"SquareLattice": SquareLattice,
                   "CheckerboardLattice": CheckerboardLattice,
                   "RectangularUnitcell": RectangularUnitcell,
                   "TriangularLattice": TriangularLattice}
class Lattice():

    def __init__(self, geometry, objects=None):
        r"""
        A dataclass combining a geometry with data container pointing to unique lattice sites.

        Parameters
        ----------
        geometry: SquareLattice
            Specify lattice geometry.
        """
        self.geometry = geometry.geometry if hasattr(geometry, 'geometry') else geometry
        for name in ["dims", "sites", "nn_site", "bonds", "site2index", "Nx", "Ny", "boundary", "f_ordered", "nn_bond_dirn"]:
            setattr(self, name, getattr(geometry, name))
        self._site_data = {self.site2index(site): None for site in self.sites()}
        self._patch = {}

        if objects is not None:
            try:
                if isinstance(objects, Sequence):
                    objects = {(nx, ny): tensor for nx, row in enumerate(objects) for ny, tensor in enumerate(row)}
                if not isinstance(objects, dict):
                    objects = {site: objects for site in self.geometry.sites()}
                for site, tensor in objects.items():
                    if self[site] is None:
                        self[site] = tensor
                    elif self[site] is not tensor:
                        raise YastnError(f"{type(self).__name__}: Non-unique assignment to unique lattice sites.")
            except (KeyError, TypeError):
                raise YastnError(f"{type(self).__name__}: Assignment outside of the lattice geometry.")
            if any(tensor is None for tensor in self._site_data.values()):
                raise YastnError(f"{type(self).__name__}: Not all unique lattice sites got assigned.")

    def __getitem__(self, site):
        """ Get tensor for site. """
        if site in self._patch:
            return self._patch[site]
        return self._site_data[self.site2index(site)]

    def __setitem__(self, site, obj):
        """ Set tensor at site. """
        if site in self._patch:
            self._patch[site] = obj
        else:
            self._site_data[self.site2index(site)] = obj

    def apply_patch(self):
        """
        Move the tensors in the patch into site_data.
        For periodic lattice, this repeats those tensors across the lattice.
        """
        for site in list(self._patch.keys()):
            self._site_data[self.site2index(site)] = self._patch.pop(site)

    def move_to_patch(self, sites):
        """ Initialize a patch with a shallow copy of data object for provided sites. """
        if not sites:
            return
        if is_site(sites):
            sites = [sites]
        for site in sites:
            self._patch[site] = self[site].shallow_copy()

    def items(self):
        """ Allows iterating over lattice sites like in dict. """
        return ((site, self[site]) for site in self.sites())

    def to_dict(self, level=2, resolve_ops=False) -> dict:
        """
        Serialize Lattice or Peps into a dictionary.
        Complementary functions are :meth:`yastn.Lattice.from_dict` and :meth:`yastn.Peps.from_dict`,
        or a general :meth:`yastn.from_dict`.
        See :meth:`yastn.Tensor.to_dict` for further description.
        """
        return {'type': type(self).__name__,
                'dict_ver': 1,
                'geometry': self.geometry.to_dict(),
                'site_data': {k: v.to_dict(level=level, resolve_ops=resolve_ops) for k, v in self._site_data.items() if v is not None}}

    @classmethod
    def from_dict(cls, d, config=None):
        r"""
        De-serializes Lattice or :class:`yastn.tn.fpeps.Peps` from the dictionary ``d``.
        See :meth:`yastn.Tensor.from_dict` for further description.
        """
        if 'dict_ver' not in d:  # d from a legacy method save_to_dict
            if 'lattice' in d:
                d['type'] = d['lattice']  # for backward compatibility
            if d['type'] in ["square", "SquareLattice"]:
                net = SquareLattice(dims=d['dims'], boundary=d['boundary'])
            elif d['type'] in ["checkerboard", "CheckerboardLattice"]:
                net = CheckerboardLattice()
            elif d['type'] in ["rectangularunitcell", "RectangularUnitcell"]:
                net = RectangularUnitcell(pattern=d['pattern'])
            elif d['type'] in ["triangular", "TriangularLattice"]:
                net = TriangularLattice()
            psi = cls(net)
            for site in psi.sites():
                obj = DATA_CLASSES["Tensor"].from_dict(d['data'][site], config)
                if obj.ndim == 3:  obj = obj.unfuse_legs(axes=(0, 1))  # for backward compatibility
                psi[site] = obj
            return psi

        if d['dict_ver'] == 1:  # d from method to_dict (single version as of now)
            if cls.__name__ != d['type']:
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
            geometry = LATTICE_CLASSES[d['geometry']['type']](**d['geometry'])
            net = cls(geometry)
            for k, v in d['site_data'].items():
                net._site_data[k] = DATA_CLASSES[v['type']].from_dict(v, config=config)
            return net

    def save_to_dict(self) -> dict:
        """
        Serialize PEPS into a dictionary.

        !!! This method is deprecated; use to_dict() instead. !!!

        """
        warn('This method is deprecated; use to_dict() instead.', DeprecationWarning, stacklevel=2)
        d = {**self.geometry.to_dict(),
             'data': {}}
        d.pop('dict_ver')
        for site in self.sites():
            d['data'][site] = self[site].save_to_dict()
        return d

    def to(self, device:str=None, dtype:str=None, **kwargs):
        r"""
        Move all PEPS tensors to specified device and/or change their data type.

        Parameters
        ----------
        device: str
            Target device.
        dtype: str
            Target data type.
        """
        net = type(self)(geometry=self.geometry)
        for ind in self._site_data:
            if self._site_data[ind] is not None:
                net._site_data[ind] = self._site_data[ind].to(device=device, dtype=dtype, **kwargs)
        return net

    def clone(self) -> Lattice:
        r"""
        Returns a deep clone of the PEPS instance by :meth:`cloning<yastn.Tensor.clone>` each tensor in
        the network. Each tensor in the cloned PEPS will contain its own independent data blocks.
        """
        net = type(self)(geometry=self.geometry)
        for ind in self._site_data:
            if self._site_data[ind] is not None:
                net._site_data[ind] = self._site_data[ind].clone()
        return net

    def copy(self) -> Lattice:
        r"""
        Returns a deep copy of the PEPS instance by :meth:`copy<yastn.Tensor.copy>` each tensor in
        the network. Each tensor in the copied PEPS will contain its own independent data blocks.
        """
        net = type(self)(geometry=self.geometry)
        for ind in self._site_data:
            if self._site_data[ind] is not None:
                net._site_data[ind] = self._site_data[ind].copy()
        return net

    def shallow_copy(self) -> Lattice:
        r"""
        New instance of :class:`yastn.tn.peps.Lattice` pointing to the same tensors as the old one.

        Shallow copy is usually sufficient to retain the old PEPS.
        """
        net = type(self)(geometry=self.geometry)
        for ind in self._site_data:
            net._site_data[ind] = self._site_data[ind]
        return net

    def detach(self) -> Lattice:
        r"""
        Return a detached view of the environment - resulting environment is **not** a part
        of the computational graph. Data of detached environment tensors is shared
        with the originals.
        """
        net = type(self)(geometry=self.geometry)
        for ind in self._site_data:
            if self._site_data[ind] is not None:
                net._site_data[ind] = self._site_data[ind].detach()
        return net

    def detach_(self):
        r"""
        Detach all environment tensors from the computational graph.
        Data of environment tensors in detached environment is a `view` of the original data.
        """
        for ind in self._site_data:
            if self._site_data[ind] is not None:
                self._site_data[ind].detach_()

    def allclose(self, other, rtol=1e-13, atol=1e-13):
        if not isinstance(other, type(self)):
            return False
        if self.geometry != other.geometry:
            return False
        for k, a in self._site_data.items():
            b = other._site_data[k]
            if (a is not None and b is not None and not a.allclose(b, rtol=rtol, atol=atol)) or \
               (a is not None and b is None) or (a is None and b is not None):
                return False
        return True

    def are_independent(self, other, independent=True):
        """
        Test if corresponding data fields have independent tensors

        independent allows testing case when all elements are None
        """
        tests = []
        for k, a in self._site_data.items():
            b = other._site_data[k]
            if a is not None and b is not None:
                tests.append(a.are_independent(b, independent=independent))
        return all(tests)
