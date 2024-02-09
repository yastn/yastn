from ... import tensordot, leg_outer_product, YastnError
from .envs._env_auxlliary import append_vec_tl, append_vec_br, append_vec_tr, append_vec_bl


_allowed_transpose = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2),
                      (0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))

class DoublePepsTensor:
    def __init__(self, top, btm, transpose=(0, 1, 2, 3)):
        r"""Class that treats a cell of a double-layer peps as a single tensor.
        Parameters
        ----------
        top: yastn.Tensor
            The top tensor of the cell.
        btm: yastn.Tensor
            The bottom tensor of the cell.
        transpose: tuple[int, int  int, int]
            Transposition with respect to canonical order of PEPS legs.
        """
        self.A = top
        self.Ab = btm
        transpose = tuple(transpose)
        if transpose not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        self._t = transpose


    @property
    def config(self):
        return self.A.config

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
        lts = self.A.get_legs(axes=axes)
        lbs = self.Ab.get_legs(axes=axes)
        legs = tuple(leg_outer_product(lt, lb.conj()) for lt, lb in zip(lts, lbs))
        return legs if multiple_legs else legs[0]


    def transpose(self, axes):
        axes = tuple(self._t[ax] for ax in axes)
        if axes not in _allowed_transpose:
            raise YastnError("DoublePEPSTensor only supports permutations that retain legs' ordering.")
        return DoublePepsTensor(self.A, self.Ab, transpose=axes)

    def conj(self):
        r""" conj """
        return DoublePepsTensor(self.A.conj(), self.Ab.conj(), transpose=self._t)

    def clone(self):
        r"""
        Makes a clone of yastn.tn.fpeps.DoublePepsTensor by :meth:`cloning<yastn.Tensor.clone>`
        all :class:`yastn.Tensor<yastn.Tensor>`'s into a new and independent :class:`peps.DoublePepsTensor`.

        .. note::
            Cloning preserves autograd tracking on all tensors.

        """
        return DoublePepsTensor(self.A.clone(), self.Ab.clone(), transpose=self._t)

    def copy(self):
        r"""
        Makes a copy of yastn.tn.fpeps.DoublePepsTensor by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.mps.MpsMpoOBC`.

        .. warning::
            this operation does not preserve autograd on the returned :code:`yastn.tn.mps.MpsMpoOBC`.

        .. note::
            Use when retaining "old" DoublePepsTensor is necessary.

        """
        return DoublePepsTensor(self.A.copy(), self.Ab.copy(), transpose=self._t)

    def append_a_bl(self, tt):
        """ Append the A and Ab tensors of self to the bottom-left corner, tt. """
        A = self.A.fuse_legs(axes=((0, 1), (2, 3), 4))
        Ab = self.Ab.fuse_legs(axes=((0, 1), (2, 3), 4))
        return append_vec_bl(A, Ab, tt)


    def append_a_tr(self, tt):
        """ Append the A and Ab tensors of self to the top-right corner, tt. """
        A = self.A.fuse_legs(axes=((0, 1), (2, 3), 4))
        Ab = self.Ab.fuse_legs(axes=((0, 1), (2, 3), 4))
        return append_vec_tr(A, Ab, tt)


    def append_a_tl(self, tt):
        """ Append the A and Ab tensors of self to the top-left corner, tt. """
        A = self.A.fuse_legs(axes=((0, 1), (2, 3), 4))
        Ab = self.Ab.fuse_legs(axes=((0, 1), (2, 3), 4))
        return append_vec_tl(A, Ab, tt)


    def append_a_br(self, tt):
        """ Append the A and Ab tensors of self to the bottom-right corner, tt. """
        A = self.A.fuse_legs(axes=((0, 1), (2, 3), 4))
        Ab = self.Ab.fuse_legs(axes=((0, 1), (2, 3), 4))
        return append_vec_br(A, Ab, tt)


    def _attach_01(self, tt):
        """
        Attach a tensor to the top left corner of the tensor network tt if rotation = 0
        and to the bottom left if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return self.append_a_tl(tt)
        if self._t == (1, 2, 3, 0):
            return self.append_a_bl(tt)
        if self._t == (2, 3, 0, 1):
            return self.append_a_br(tt)
        if self._t == (3, 0, 1, 2):
            return self.append_a_tr(tt)
        if self._t == (0, 3, 2, 1):
            return self.append_a_tr(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (3, 2, 1, 0):
            return self.append_a_br(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (2, 1, 0, 3):
            return self.append_a_bl(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (1, 0, 3, 2):
            return self.append_a_tl(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))


    def _attach_23(self, tt):
        """
        Attach a tensor to the bottom right corner of the tensor network tt if rotation = 0
        and to the top right if rotation is 90.
        """
        if self._t == (0, 1, 2, 3):
            return self.append_a_br(tt)
        if self._t == (1, 2, 3, 0):
            return self.append_a_tr(tt)
        if self._t == (2, 3, 0, 1):
            return self.append_a_tl(tt)
        if self._t == (3, 0, 1, 2):
            return self.append_a_bl(tt)
        if self._t == (0, 3, 2, 1):
            return self.append_a_bl(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (3, 2, 1, 0):
            return self.append_a_tl(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (2, 1, 0, 3):
            return self.append_a_tr(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))
        if self._t == (1, 0, 3, 2):
            return self.append_a_br(tt.transpose((0, 2, 1, 3))).transpose((0, 3, 2, 1))


    def fPEPS_fuse_layers(self):
        """
        Fuse the top and bottom layers of a PEPS tensor network.
        """

        fA = self.top.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0t 1t) (2t 3t) 4t
        fAb = self.btm.fuse_legs(axes=((0, 1), (2, 3), 4))  # (0b 1b) (2b 3b) 4b
        tt = tensordot(fA, fAb, axes=(2, 2), conj=(0, 1))  # (0t 1t) (2t 3t) (0b 1b) (2b 3b)
        tt = tt.fuse_legs(axes=(0, 2, (1, 3)))  # (0t 1t) (0b 1b) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=(0, 1))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.swap_gate(axes=(1, 2))  # 0t 1t 0b 1b ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((0, 2), (1, 3), 4))  # (0t 0b) (1t 1b) ((2t 3t) (2b 3b))
        tt = tt.fuse_legs(axes=((1, 0), 2))  # ((1t 1b) (0t 0b)) ((2t 3t) (2b 3b))
        tt = tt.unfuse_legs(axes=1)  # ((1t 1b) (0t 0b)) (2t 3t) (2b 3b)
        tt = tt.unfuse_legs(axes=(1, 2))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.swap_gate(axes=(1, 4))  # ((1t 1b) (0t 0b)) 2t 3t 2b 3b
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))  # ((1t 1b) (0t 0b)) (2t 2b) (3t 3b)
        st = tt.unfuse_legs(axes=(0)) # (1t 1b) (0t 0b) (3t 3b) (2t 2b)
        st = st.fuse_legs(axes=(1, 0, 3, 2)) # (0t 0b) (1t 1b) (2t 2b) (3t 3b)
        return st
