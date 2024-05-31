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
""" Initialization of peps tensors for real or imaginary time evolution """
from ._geometry import SquareLattice, CheckerboardLattice
from ._peps import Peps
from .envs._env_ctm import EnvCTM, EnvCTM_local
from ...initialize import load_from_dict as load_tensor_from_dict
from ... import YastnError, Tensor


def product_peps(geometry, vectors) -> Peps:
    """
    Initialize PEPS in a product state composed of provided vectors.

    Vectors can have ndim=1 for (pure) state and ndim=2 for purification/operator.
    In the latter case, two legs will be fused into one physical leg.
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
    if not isinstance(geometry, (SquareLattice, CheckerboardLattice)):
        raise YastnError("Geometry should be an instance of SquareLattice or CheckerboardLattice")

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
        raise YastnError("product_peps did not initialize some peps tensor")
    return psi


def load_from_dict(config, d) -> Peps:
    r"""
    Create PEPS from dictionary.

    Parameters
    ----------
    config: module | _config(NamedTuple)
        :ref:`YASTN configuration <tensor/configuration:yastn configuration>`

    in_dict: dict
        dictionary containing serialized MPS/MPO, i.e.,
        a result of :meth:`yastn.tn.mps.MpsMpo.save_to_dict`.
    """

    if 'class' in d and d['class'] == 'EnvCTM':  # load EnvCTM
        psi = load_from_dict(config, d['psi'])
        env = EnvCTM(psi, init=None)
        for site in env.sites():
            for dirn, v in d['data'][site].items():
                setattr(env[site], dirn, load_tensor_from_dict(config, v))
        return env

    # otherwise assume class == 'Peps'
    if d['lattice'] == "square":
        net = SquareLattice(dims=d['dims'], boundary=d['boundary'])
    elif d['lattice'] == "checkerboard":
        net = CheckerboardLattice()

    psi = Peps(net)
    for site in psi.sites():
        psi[site] = load_tensor_from_dict(config, d['data'][site])
    return psi
