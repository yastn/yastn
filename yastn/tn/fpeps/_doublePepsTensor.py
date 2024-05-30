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
from ... import tensordot, leg_outer_product, YastnError
from .envs._env_auxlliary import append_vec_tl, append_vec_br, append_vec_tr, append_vec_bl


_allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                      (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))

class DoublePepsTensor:
    def __init__(self, top, btm, transpose=(0, 1, 2, 3)):
        r"""
        Class that treats a cell of a double-layer peps as a single tensor.

        Parameters
        ----------
        top: yastn.Tensor
            The top tensor of the cell.
        btm: yastn.Tensor
            The bottom tensor of the cell.
        transpose: tuple[int, int  int, int]
            Transposition with respect to canonical order of PEPS legs.
        """
        self.top = top
        self.btm = btm
        transpose = tuple(transpose)
        if transpose not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        self._t = transpose

    @property
    def config(self):
        return self.top.config

    @property
    def ndim(self):
        return 4

    def get_shape(self, axes=None):
        """ Returns the shape of the DoublePepsTensor along the specified axes """
        if axes is None:
            axes = tuple(range(4))
        if isinstance(axes, int):
            return sum(self.get_legs(axes).D)
        return tuple(sum(leg.D) for leg in self.get_legs(axes))

    def get_legs(self, axes=None):
        """ Returns the legs of the DoublePepsTensor along the specified axes. """
        if axes is None:
            axes = tuple(range(4))
        multiple_legs = hasattr(axes, '__iter__')
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        axes = tuple(self._t[ax] for ax in axes)

        lts = self.top.get_legs(axes=(0, 1))
        lbs = self.btm.get_legs(axes=(0, 1))
        lts = [*lts[0].unfuse_leg(), *lts[1].unfuse_leg()]
        lbs = [*lbs[0].unfuse_leg(), *lbs[1].unfuse_leg()]
        lts = [lts[i] for i in axes]
        lbs = [lbs[i] for i in axes]
        # lts = self.top.get_legs(axes=axes)
        # lbs = self.btm.get_legs(axes=axes)
        legs = tuple(leg_outer_product(lt, lb.conj()) for lt, lb in zip(lts, lbs))
        return legs if multiple_legs else legs[0]

    def transpose(self, axes):
        axes = tuple(self._t[ax] for ax in axes)
        if axes not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        return DoublePepsTensor(self.top, self.btm, transpose=axes)

    def conj(self):
        r""" conj """
        return DoublePepsTensor(self.top.conj(), self.btm.conj(), transpose=self._t)

    def clone(self):
        r"""
        Makes a clone of yastn.tn.fpeps.DoublePepsTensor by :meth:`cloning<yastn.Tensor.clone>`-ing
        all constituent tensors forming a new instance of :class:`peps.DoublePepsTensor`.
        """
        return DoublePepsTensor(self.top.clone(), self.btm.clone(), transpose=self._t)

    def copy(self):
        r"""
        Makes a copy of yastn.tn.fpeps.DoublePepsTensor by :meth:`copying<yastn.Tensor.copy>`-ing
        all constituent tensors forming a new instance of :class:`peps.DoublePepsTensor`.
        """
        return DoublePepsTensor(self.top.copy(), self.btm.copy(), transpose=self._t)

    def _attach_01(self, tt):
        """
        Attach a tensor to the top left corner of the tensor network tt if rotation = 0
        and to the bottom left if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return append_vec_tl(self.top, self.btm, tt)
        if self._t == (1, 2, 3, 0):
            return append_vec_bl(self.top, self.btm, tt)
        if self._t == (2, 3, 0, 1):
            return append_vec_br(self.top, self.btm, tt)
        if self._t == (3, 0, 1, 2):
            return append_vec_tr(self.top, self.btm, tt)
        if self._t == (0, 3, 2, 1):
            return append_vec_tr(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (3, 2, 1, 0):
            return append_vec_br(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (2, 1, 0, 3):
            return append_vec_bl(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (1, 0, 3, 2):
            return append_vec_tl(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))

    def _attach_23(self, tt):
        """
        Attach a tensor to the bottom right corner of the tensor network tt if rotation = 0
        and to the top right if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return append_vec_br(self.top, self.btm, tt)
        if self._t == (1, 2, 3, 0):
            return append_vec_tr(self.top, self.btm, tt)
        if self._t == (2, 3, 0, 1):
            return append_vec_tl(self.top, self.btm, tt)
        if self._t == (3, 0, 1, 2):
            return append_vec_bl(self.top, self.btm, tt)
        if self._t == (0, 3, 2, 1):
            return append_vec_bl(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (3, 2, 1, 0):
            return append_vec_tl(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (2, 1, 0, 3):
            return append_vec_tr(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (1, 0, 3, 2):
            return append_vec_br(self.top, self.btm, tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))

    def _attach_30(self, tt):
        """
        Attach a tensor to the top left corner of the tensor network tt if rotation = 0
        and to the bottom left if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return append_vec_tr(self.top, self.btm, tt)
        raise YastnError(f'Transpositions not supported by _attach_30')

    def _attach_12(self, tt):
        """
        Attach a tensor to the top left corner of the tensor network tt if rotation = 0
        and to the bottom left if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return append_vec_bl(self.top, self.btm, tt)
        raise YastnError(f'Transpositions not supported by _attach_12')

    def fuse_layers(self):
        """
        Fuse the top and bottom layers of a PEPS tensor network.
        """
        tt = tensordot(self.top, self.btm, axes=(2, 2), conj=(0, 1))  # [t l] [b r] [t' l'] [b' r']
        tt = tt.fuse_legs(axes=(0, 2, (1, 3)))  # [t l] [t' l'] [[b r] [b' r']]
        tt = tt.unfuse_legs(axes=(0, 1))  # t l t' l' [[b r] [b' r']]
        tt = tt.swap_gate(axes=((1, 3), 2))  # l l' X t'
        tt = tt.fuse_legs(axes=((0, 2), (1, 3), 4))  # [t t'] [l l'] [[b r] [b' r']]
        tt = tt.fuse_legs(axes=((0, 1), 2))  # [[t t'] [l l']] [[b r] [b' r']]
        tt = tt.unfuse_legs(axes=1)  # [[t t'] [l l']] [b r] [b' r']
        tt = tt.unfuse_legs(axes=(1, 2))  # [[t t'] [l l']] b r b' r'
        tt = tt.swap_gate(axes=((1, 3), 4))  # b b' X r'
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))  # [[t t'] [l l']] [b b'] [r r']
        tt = tt.unfuse_legs(axes=0) # [t t'] [l l'] [b b'] [r r']
        return tt.transpose(axes=self._t)
