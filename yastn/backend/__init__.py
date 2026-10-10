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
""" Backends and their lookup by ``BACKEND_ID``. """
from functools import lru_cache

__all__ = ['import_backend']


@lru_cache(maxsize=None)
def import_backend(backend_id):
    """
    Load a yastn backend module by its ``BACKEND_ID`` string (``'np'``, ``'torch'``,
    ``'torch_cutensor'``). Cached so repeated lookups return the same module object.
    Raises ``ValueError`` for an unknown id.
    """
    if backend_id in (None, 'np', 'numpy'):
        from . import backend_np
        return backend_np
    if backend_id == 'torch':
        from . import backend_torch
        return backend_torch
    if backend_id == 'torch_cutensor':
        from . import backend_torch_cutensor  # pragma: no cover
        return backend_torch_cutensor  # pragma: no cover
    raise ValueError("backend encoded as string only supports: 'np', 'torch', 'torch_cutensor'")
