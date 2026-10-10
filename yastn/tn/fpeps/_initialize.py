# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================
""" Initialization of PEPS tensors for real or imaginary time evolution """
from ._geometry import SquareLattice, CheckerboardLattice, RectangularUnitcell, TriangularLattice
from ._peps import Peps
from .envs._env_boundary_mps import EnvBoundaryMPS
from .envs._env_bp import EnvBP
from .envs._env_ctm import EnvCTM
from ...tensor import YastnError, Tensor


def product_peps(geometry, vectors) -> Peps:
    """
    Initialize PEPS in a product state composed of provided vectors for all lattice sites.

    Vectors can have effective rank ndim=1 for (pure) state and ndim=2 for purification/operator.
    In the latter case, two legs will be fused into one physical PEPS leg.
    Virtual legs of dimension one with zero charge are added automatically.
    For vectors, their possibly non-zero charge is incorporated by
    adding an auxiliary leg with dimension one.

    Parameters
    ----------
    geometry : SquareLattice | CheckerboardLattice
        lattice geometry.
    vectors : yastn.Tensor | Dict[tuple[Int, Int], yastn.Tensor]
        If yastn.Tensor is provided, it gets repeated across the lattice.
        If dict is provided, it should specify a map between
        each unique lattice site and the corresponding vector.
    """
    if not isinstance(geometry, (SquareLattice, CheckerboardLattice, TriangularLattice, RectangularUnitcell)):
        raise YastnError("Geometry should be an instance of SquareLattice or CheckerboardLattice or TriangularLattice or RectangularUnitcell")

    if isinstance(vectors, Tensor):
        vectors = {site: vectors.copy() for site in geometry.sites()}

    for k, v in vectors.items():
        if v.ndim == 1 and not v.get_legs(axes=0).is_fused():
            v = v.add_leg(s=-1)
        if v.ndim == 2:
            v = v.fuse_legs(axes=[(0, 1)])
            vectors[k] = v
        if v.ndim > 1:
            raise YastnError("Some vector has ndim > 2")

    psi = Peps(geometry)
    for site, vec in vectors.items():
        for s in (-1, 1, 1, -1):
            vec = vec.add_leg(axis=0, s=s)
        psi[site] = vec
    if any(psi[site] is None for site in psi.sites()):
        raise YastnError("product_peps did not initialize some PEPS tensor")
    return psi
