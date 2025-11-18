# Copyright 2025 The YASTN Authors. All Rights Reserved.
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
from dataclasses import dataclass, fields

from ....tensor import Tensor, YastnError

__all__ = ["EnvBP_local", "EnvCTM_local", "EnvCTM_projectors", "EnvCTM_c4v_local", "EnvCTM_c4v_projectors", "Gauge"]


@dataclass()
class dataclasses_common():

    def to(self, device: str=None, dtype: str=None):
        return type(self)(**{k.name: getattr(self, k.name).to(device=device, dtype=dtype) for k in fields(self)
                             if getattr(self, k.name) is not None})

    def copy(self):
        return type(self)(**{k.name: getattr(self, k.name).copy() for k in fields(self)
                             if getattr(self, k.name) is not None})

    def clone(self):
        return type(self)(**{k.name: getattr(self, k.name).clone() for k in fields(self)
                             if getattr(self, k.name) is not None})

    def detach(self):
        return type(self)(**{k.name: getattr(self, k.name).detach() for k in fields(self)
                             if getattr(self, k.name) is not None})

    def shallow_copy(self):
        return type(self)(**{k.name: getattr(self, k.name) for k in fields(self)
                             if getattr(self, k.name) is not None})

    def detach_(self):
        for k in fields(self):
            if getattr(self, k.name) is not None:
                getattr(self, k.name).detach_()

    def allclose(self, other, rtol=1e-13, atol=1e-13):
        if type(self) != type(other):
            return False
        for k in fields(self):
            a, b = getattr(self, k.name), getattr(other, k.name)
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
        for k in fields(self):
            ta, tb = getattr(self, k.name), getattr(other, k.name)
            if ta is not None and tb is not None:
                tests.append(ta.are_independent(tb) == independent)
        return all(tests)

    def to_dict(self, level=2):
        """ Return a dictionary representation of the object. """
        d = {'type': type(self).__name__,
             'dict_ver': 1}
        for k in fields(self):
            if getattr(self, k.name) is not None:
                d[k.name] = getattr(self, k.name).to_dict(level=level)
        return d

    @classmethod
    def from_dict(cls, d, config=None):
        r""" De-serializes dataclass object from the dictionary ``d``. """
        if d['dict_ver'] == 1:
            if cls.__name__ != d['type']:
                raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
            dd = {k.name: Tensor.from_dict(d[k.name], config=config) for k in fields(cls) if k.name in d}
            return cls(**dd)

    def __repr__(self) -> str:
        return f"{type(self)}({ ',\n'.join(f'{k}={v}' for k,v in self.__dict__.items()) })"

@dataclass()
class EnvCTM_local(dataclasses_common):
    r"""
    Dataclass for CTM environment tensors associated with Peps lattice site.
    Contains fields ``tl``, ``t``, ``tr``, ``r``, ``br``, ``b``, ``bl``, ``l``
    """
    tl: Tensor | None = None  # top-left
    t:  Tensor | None = None  # top
    tr: Tensor | None = None  # top-right
    r:  Tensor | None = None  # right
    br: Tensor | None = None  # bottom-right
    b:  Tensor | None = None  # bottom
    bl: Tensor | None = None  # bottom-left
    l:  Tensor | None = None  # left


@dataclass()
class EnvCTM_projectors(dataclasses_common):
    r"""
    Dataclass for CTM projectors associated with Peps lattice site.
    """
    hlt: Tensor | None = None  # horizontal left top
    hlb: Tensor | None = None  # horizontal left bottom
    hrt: Tensor | None = None  # horizontal right top
    hrb: Tensor | None = None  # horizontal right bottom
    vtl: Tensor | None = None  # vertical top left
    vtr: Tensor | None = None  # vertical top right
    vbl: Tensor | None = None  # vertical bottom left
    vbr: Tensor | None = None  # vertical bottom right


@dataclass()
class EnvBP_local(dataclasses_common):
    r"""
    Dataclass for BP environment tensors at a single Peps site on square lattice.
    Contains fields ``t``,  ``l``, ``b``, ``r``
    """
    t: Tensor | None = None  # top R.H @ R
    l: Tensor | None = None  # left R.H @ R
    b: Tensor | None = None  # bottom R.H @ R
    r: Tensor | None = None  # right R.H @ R
    tR: Tensor | None = None  # top R
    lR: Tensor | None = None  # left R
    bR: Tensor | None = None  # bottom R
    rR: Tensor | None = None  # right R

    def __getattribute__(self, dirn):
        if dirn in ['t', 'l', 'b', 'r']:
            res = super(EnvBP_local, self).__getattribute__(dirn)
            if res is None:
                resR = super(EnvBP_local, self).__getattribute__(dirn + "R")
                if resR is not None:
                    res = resR.H @ resR
                    super(EnvBP_local, self).__setattr__(dirn, res)
            return res
        else:
            return super(EnvBP_local, self).__getattribute__(dirn)


@dataclass()
class EnvCTM_c4v_local(dataclasses_common):
    r"""
    Dataclass for CTM environment tensors associated with Peps lattice site.
    Contains fields ``tl``, ``t``, ``tr``, ``r``, ``br``, ``b``, ``bl``, ``l``
    """
    tl: Tensor | None = None  # top-left
    t:  Tensor | None = None  # top

    def __getattr__(self, dirn):
        if dirn in ["tr", "br", "bl"]:
            return self.tl
        if dirn in ["l", "b", "r"]:
            return self.t
        raise AttributeError()

@dataclass()
class EnvCTM_c4v_projectors(dataclasses_common):
    r"""
    Dataclass for CTM projectors associated with Peps lattice site.
    """
    vtl: Tensor | None = None  # vertical top left
    vtr: Tensor | None = None  # vertical top right

@dataclass()
class Gauge(dataclasses_common):
    r"""
    Dataclass for gauge matrices at a single Peps site on square lattice.
    Contains fields ``t``,  ``l``, ``b``, ``r``
    """
    t: Tensor | None = None  # top
    l: Tensor | None = None  # left
    b: Tensor | None = None  # bottom
    r: Tensor | None = None  # right

DATA_CLASSES = {"Tensor": Tensor,
                "EnvBP_local": EnvBP_local,
                "EnvCTM_local": EnvCTM_local,
                "EnvCTM_projectors": EnvCTM_projectors,
                "EnvCTM_c4v_local": EnvCTM_c4v_local,
                "EnvCTM_c4v_projectors": EnvCTM_c4v_projectors,
                "Gauge": Gauge}
