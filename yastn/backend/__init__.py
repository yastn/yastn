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
