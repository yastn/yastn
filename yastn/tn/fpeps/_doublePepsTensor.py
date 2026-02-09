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
from __future__ import annotations
from typing import Sequence

from ._gates_auxiliary import match_ancilla, apply_gate_onsite
from .envs._env_contractions import append_vec_tl, append_vec_br, append_vec_tr, append_vec_bl
from ...tensor import tensordot, leg_product, YastnError, SpecialTensor, Tensor
from ...tensor._auxliary import _clear_axes


_allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                      (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))


class DoublePepsTensor(SpecialTensor):
    def __init__(self, bra, ket, transpose=(0, 1, 2, 3), op=None, swaps=None):
        r"""
        Class that treats a pair of tensors forming a site of double-layer PEPS as a single tensor.

        Parameters
        ----------
        bra: yastn.Tensor
            The "bra" (or bottom) tensor of the cell.
        ket: yastn.Tensor
            The "ket" (or top) tensor of the cell.
        transpose: tuple[int, int  int, int]
            Transposition with respect to canonical order of PEPS legs.
        """
        self.bra = bra
        self.op = op
        self.ket = ket
        transpose = tuple(transpose)
        if transpose not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        self._t = transpose
        self.swaps = {} if swaps is None else dict(swaps)

    @property
    def config(self):
        return self.ket.config

    @property
    def ndim(self):
        return 4

    def set_operator_(self, op, reset=True):
        """
        Include the operator that is applied on the physical leg of the ket tensor during contraction.

        By default, it resets the previous operator (if present).
        Otherwise, multiply the previous operator from the left, i.e., apply it after the one in self.op.
        """
        op = match_ancilla(self.ket, op)
        self.op = op if (reset or self.op is None) else op @ self.op

    def del_operator_(self):
        """ Remove operator. """
        self.op = None

    def apply_gate_on_ket(self, op, dirn):
        """ Returns a shallow copy with ket tensor modified by application of gate. """
        if 'k4' in self.swaps:
            op = op.swap_gate(axes=2, charge=self.swaps.pop('k4'))
        ket = apply_gate_onsite(self.ket, op, dirn=dirn)
        return DoublePepsTensor(bra=self.bra, ket=ket, transpose=self._t, op=self.op, swaps=self.swaps)

    def add_charge_swaps_(self, charge, axes):
        """
        Supplement DoublePepsTensor with charges to be swapped with some internal legs during contraction.

        Parameters
        ----------
        charge: tuple[int]

        axes: Sequence[str] | str
            identfy axes: 'bt', 'bl', 'bb', 'br', 'bs', 'kt', 'kl', 'kb', 'kr', 'ks'
            k/b is for ket/bra; t/l/b/r/s is for top/left/bottom/right/system
        """
        if isinstance(axes, str):
            axes = [axes]
        for ax in axes:
            if ax not in ['b0', 'b1', 'b2', 'b3', 'b4', 'k0', 'k1', 'k2', 'k3', 'k4']:
                raise YastnError("Elements of axes should be 'b0', 'b1', 'b2', 'b3', 'b4', 'k0', 'k1', 'k2', 'k3', 'k4'.")
            t = self.swaps.pop(ax, None)
            t = self.config.sym.add_charges(t, charge) if t is not None else charge
            if t != self.config.sym.zero():
                self.swaps[ax] = t

    def del_charge_swaps_(self):
        """ Remove all charge swaps. """
        self.swaps = {}

    def has_operator_or_swap(self):
        return self.op is not None or (self.config.fermionic and self.swaps)

    def Ab_Ak_with_charge_swap(self):
        if not self.swaps:
            return self.bra, self.ket
        Ab = self.bra
        axes, charges = [], []
        for ax, n in self.swaps.items():
            if ax[0] == 'b':
                axes.append(int(ax[1]))
                charges.append(n)
        if axes:
            Ab = Ab.swap_gate(axes, charge=charges)

        Ak = self.ket
        axes, charges = [], []
        for ax, n in self.swaps.items():
            if ax[0] == 'k':
                axes.append(int(ax[1]))
                charges.append(n)
        if axes:
            Ak = Ak.swap_gate(axes, charge=charges)
        return Ab, Ak

    def get_shape(self, axes=None):
        """ Returns the shape of the DoublePepsTensor along specified axes. """
        if axes is None:
            axes = tuple(range(4))
        if isinstance(axes, int):
            return sum(self.get_legs(axes).D)
        return tuple(sum(leg.D) for leg in self.get_legs(axes))

    @property
    def shape(self):
        return self.get_shape()

    def get_legs(self, axes=None):
        """ Returns the legs of the DoublePepsTensor along specified axes. """
        if axes is None:
            axes = tuple(range(4))
        multiple_legs = hasattr(axes, '__iter__')
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        axes = tuple(self._t[ax] for ax in axes)

        lts = self.ket.get_legs(axes=(0, 1, 2, 3))
        lbs = self.bra.get_legs(axes=(0, 1, 2, 3))
        lts = [lts[i] for i in axes]
        lbs = [lbs[i] for i in axes]
        # lts = self.ket.get_legs(axes=axes)
        # lbs = self.bra.get_legs(axes=axes)
        legs = tuple(leg_product(lt, lb.conj()) for lt, lb in zip(lts, lbs))
        return legs if multiple_legs else legs[0]

    def get_signature(self):
        return self.s

    @property
    def s(self) -> Sequence[int]:
        return self.ket.s[:4]

    def transpose(self, axes):
        """ Transposition of DoublePepsTensor. Only cyclic permutations are allowed. """
        axes = tuple(self._t[ax] for ax in axes)
        if axes not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        return DoublePepsTensor(bra=self.bra, ket=self.ket, transpose=axes, op=self.op, swaps=self.swaps)

    # def flip_signature(self):
    #     r""" Conjugate DoublePepsTensor. """
    #     op_fs = self.op.flip_signature() if self.op is not None else None
    #     return DoublePepsTensor(bra=self.bra.flip_signature(), ket=self.ket.flip_signature(), transpose=self._t, op=op_fs, swaps=self.swaps)

    def conj(self):
        r""" Conjugate DoublePepsTensor. """
        op_conj = self.op.conj() if self.op is not None else None
        return DoublePepsTensor(bra=self.bra.conj(), ket=self.ket.conj(), transpose=self._t, op=op_conj, swaps=self.swaps)

    def to(self, device=None, dtype=None, **kwargs):
        r"""
        Move DoublePepsTensor to device and cast to given datatype.

        Returns a clone of the DoublePepsTensor residing on ``device`` in desired datatype ``dtype``.
        If DoublePepsTensor already resides on ``device``, returns ``self``. This operation preserves autograd.
        If no change is needed, makes only a shallow copy of the tensor data.

        Parameters
        ----------
        device: str
            device identifier
        dtype: str
            desired dtype
        """
        op_clone = self.op.to(device=device, dtype=dtype, **kwargs) if self.op is not None else None
        return DoublePepsTensor(bra=self.bra.to(device=device, dtype=dtype, **kwargs), \
                                ket=self.ket.to(device=device, dtype=dtype, **kwargs), transpose=self._t, op=op_clone, swaps=self.swaps)

    def clone(self):
        r"""
        Makes a clone of yastn.tn.fpeps.DoublePepsTensor by :meth:`cloning<yastn.Tensor.clone>`-ing
        all constituent tensors forming a new instance of DoublePepsTensor.
        """
        op_clone = self.op.clone() if self.op is not None else None
        return DoublePepsTensor(bra=self.bra.clone(), ket=self.ket.clone(), transpose=self._t, op=op_clone, swaps=self.swaps)

    def copy(self):
        r"""
        Makes a copy of yastn.tn.fpeps.DoublePepsTensor by :meth:`copying<yastn.Tensor.copy>`-ing
        all constituent tensors forming a new instance of DoublePepsTensor.
        """
        op_copy = self.op.copy() if self.op is not None else None
        return DoublePepsTensor(bra=self.bra.copy(), ket=self.ket.copy(), transpose=self._t, op=op_copy, swaps=self.swaps)

    def to_dict(self, level=2):
        r""" Serialize DoublePepsTensor into a dictionary. """
        d = {'type': type(self).__name__,
             'dict_ver': 1,
             'bra': self.bra.to_dict(level=level),
             'ket': self.ket.to_dict(level=level),
             'transpose': self._t,
             'swaps': self.swaps.copy()}
        if self.op is not None:
            d['op'] = self.op.to_dict(level=level)
        return d

    @classmethod
    def from_dict(cls, d, config=None) -> DoublePepsTensor:
        r"""
        De-serializes DoublePepsTensor from the dictionary ``d``.
        See :meth:`yastn.Tensor.from_dict` for further description.
        """
        if cls.__name__ != d['type']:
            raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
        bra = Tensor.from_dict(d=d['bra'], config=config)
        ket = Tensor.from_dict(d=d['ket'], config=config)
        op = Tensor.from_dict(d=d['op'], config=config) if 'op' in d else None
        return DoublePepsTensor(bra=bra, ket=ket, transpose=d['transpose'], op=op, swaps=d['swaps'])

    def tensordot(self, b, axes, reverse=False):
        r"""
        tensordot(DublePepsTensor, b, axes) with tensor leg order conventions matching the default for tensordot.
        tensordot(self, b, axes, reverse=True) corresponds to tensordot(b, self, axes).
        """

        if reverse:
            axes = axes[::-1]
            mode = "b-self"
        else:
            mode = "self-b"

        in_a, in_b = _clear_axes(*axes)  # contracted meta legs
        if len(in_a) != 2 or len(in_b) != 2:
            raise YastnError('DoublePepsTensor.tensordot only supports contraction of exactly 2 legs.')
        sa0, sa1 = set(in_a), set(in_b)
        if len(sa0) != len(in_a) or len(sa1) != len(in_b):
            raise YastnError('DoublePepsTensor.tensordot repeated axis in axes[0] or axes[1].')
        if sa0 - set(range(self.ndim)) or sa1 - set(range(b.ndim)):
            raise YastnError('DoublePepsTensor.tensordot axes outside of tensor ndim.')

        in_a = tuple(self._t[ax] for ax in in_a)
        out_a = tuple(ax for ax in self._t if ax not in in_a)

        if in_a[0] > in_a[1]:  # reference order is (t, l, b, r)
            in_a = in_a[::-1]
            in_b = in_b[::-1]

        Ab, Ak = self.Ab_Ak_with_charge_swap()

        if in_a == (0, 1):
            return append_vec_tl(Ab, Ak, b, op=self.op, mode=mode, in_b=in_b, out_a=out_a)
        elif in_a == (2, 3):
            return append_vec_br(Ab, Ak, b, op=self.op, mode=mode, in_b=in_b, out_a=out_a)
        elif in_a == (0, 3):
            return append_vec_tr(Ab, Ak, b, op=self.op, mode=mode, in_b=in_b, out_a=out_a)
        elif in_a == (1, 2):
            return append_vec_bl(Ab, Ak, b, op=self.op, mode=mode, in_b=in_b, out_a=out_a)
        raise YastnError('DoublePepsTensor.tensordot, 2 axes of self should be neighbouring.')

    def fuse_layers(self):
        """
        Fuse the top and bottom tensors into a single :class:`yastn.Tensor`.
        # """
        Ab, Ak = self.Ab_Ak_with_charge_swap()
        Ab = Ab.fuse_legs(axes=((0, 1), (2, 3), 4))  # A -> [t l] [b r] s
        Ak = Ak.fuse_legs(axes=((0, 1), (2, 3), 4))
        if self.op is not None:
            Ak = tensordot(Ak, self.op, axes=(2, 1))
        tt = tensordot(Ak, Ab.conj(), axes=(2, 2))  # [t l] [b r] [t' l'] [b' r']
        tt = tt.unfuse_legs(axes=(0, 2))  # t l [b r] t' l' [b' r']
        tt = tt.swap_gate(axes=((1, 4), 3))  # l l' X t'
        tt = tt.fuse_legs(axes=((0, 3), (1, 4), 2, 5))  # [t t'] [l l'] [b r] [b' r']
        tt = tt.unfuse_legs(axes=(2, 3))  # [t t'] [l l'] b r b' r'
        tt = tt.swap_gate(axes=((2, 4), 5))  # b b' X r'
        tt = tt.fuse_legs(axes=(0, 1, (2, 4), (3, 5)))  # [t t'] [l l'] [b b'] [r r']
        return tt.transpose(axes=self._t)

    def print_properties(self, file=None):
        """ Print basic properties of DoublePepsTensor. """
        print("DoublePepsTensor", file=file)
        print("shape   :", self.get_shape(), file=file)
        st = {i: leg.history() for i, leg in enumerate(self.get_legs())}
        print("legs fusions :", st, "\n", file=file)
