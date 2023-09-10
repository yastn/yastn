from yastn import tensordot, leg_outer_product

_rotations = {0: (0, 1, 2, 3), 90: (1, 2, 3, 0)}

class DoublePepsTensor:

    r"""Class that treats a cell of a double-layer peps as a single tensor.

    Parameters
    ----------
        top : yastn.Tensor
            The top tensor of the cell.
        btm : yastn.Tensor
            The bottom tensor of the cell.
        rotation : int, optional
            The rotation angle of the cell, default is 0.

    Attributes
    ----------
        A : yastn.Tensor
            The top tensor of the cell.
        Ab : yastn.Tensor
            The bottom tensor of the cell.
        _r : int
            The rotation angle of the cell.

    """

    def __init__(self, top, btm, rotation=0):
        self.A = top
        self.Ab = btm
        self._r = rotation

    @property
    def ndim(self):
        return 4

    def get_shape(self, axes=None):
        """ Returns the shape of the DoublePepsTensor along the specified axes """

        if axes is None:
            axes = tuple(range(4))
        sA = self.A.get_shape(axes=axes)
        sB = self.Ab.get_shape(axes=axes)
        if isinstance(axes, int):
            return sA * sB 
        return tuple(x * y for x, y in zip(sA, sB))

    def get_legs(self, axes=None):
        """ Returns the legs of the DoublePepsTensor along the specified axes. """
        
        if axes is None:
            axes = tuple(range(4))
        multiple_legs = hasattr(axes, '__iter__')
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        rot = _rotations[self._r]
        axes = tuple(rot[ax] for ax in axes)
        lts = self.A.get_legs(axes=axes)
        lbs = self.Ab.get_legs(axes=axes)
        legs = tuple(leg_outer_product(lt, lb.conj()) for lt, lb in zip(lts, lbs))
        return legs if multiple_legs else legs[0]


    def clone(self):
        r"""
        Makes a clone of yastn.tn.fpeps.DoublePepsTensor by :meth:`cloning<yastn.Tensor.clone>`
        all :class:`yastn.Tensor<yastn.Tensor>`'s into a new and independent :class:`peps.DoublePepsTensor`.

        .. note::
            Cloning preserves autograd tracking on all tensors.

        """
        return DoublePepsTensor(self.A.clone(), self.Ab.clone(), rotation=self._r)

    def copy(self):
        r"""
        Makes a copy of yastn.tn.fpeps.DoublePepsTensor by :meth:`copying<yastn.Tensor.copy>` all :class:`yastn.Tensor<yastn.Tensor>`'s
        into a new and independent :class:`yastn.tn.mps.MpsMpo`.

        .. warning::
            this operation does not preserve autograd on the returned :code:`yastn.tn.mps.MpsMpo`.

        .. note::
            Use when retaining "old" DoublePepsTensor is necessary. 

        """
        return DoublePepsTensor(self.A.copy(), self.Ab.copy(), rotation=self._r)

    def append_a_bl(self, tt):
        """ Append the A and Ab tensors of self to the bottom-left corner, tt. """

        tt = tt.fuse_legs(axes=((0, 3), 1, 2))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))
        Abf = self.Ab.fuse_legs(axes=((2, 1), (3, 0, 4)))
        Af = self.A.fuse_legs(axes=((2, 1, 4), (3, 0)))
        tt = tt.tensordot(Abf, axes=(2, 0), conj=(0, 1))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(1, 3, 2, 4))
        tt = tt.fuse_legs(axes=(0, (3, 4), (1, 2, 5)))
        tt = tt.tensordot(Af, axes=(2, 0))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.fuse_legs(axes=(0, (3, 1), (4, 2)))
        tt = tt.unfuse_legs(axes=0)
        tt = tt.transpose(axes=(0, 2, 1, 3))
        return tt

    def append_a_tr(self, tt):
        """ Append the A and Ab tensors of self to the top-right corner, tt. """

        tt = tt.fuse_legs(axes=((0, 3), 1, 2))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.fuse_legs(axes=(0, (2, 4), (1, 3)))
        Af = self.A.fuse_legs(axes=((0, 3), (1, 2, 4)))
        Abf = self.Ab.fuse_legs(axes=((0, 3, 4), (1, 2)))
        tt = tt.tensordot(Af, axes=(2, 0))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(1, 3, 2, 4))
        tt = tt.fuse_legs(axes=(0, (3, 4), (1, 2, 5)))
        tt = tt.tensordot(Abf, axes=(2, 0), conj=(0, 1))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))
        tt = tt.unfuse_legs(axes=0)
        tt = tt.transpose(axes=(0, 2, 1, 3))
        return tt

    def append_a_tl(self, tt):
        """ Append the A and Ab tensors of self to the top-left corner, tt. """

        # tt: e2 (1t 1b) (0t 0b) e3
        tt = tt.fuse_legs(axes=((0, 3), 1, 2))  # (e2 e3) (1t 1b) (0t 0b)
        tt = tt.unfuse_legs(axes=(1, 2))  # (e2 e3) 1t 1b 0t 0b
        tt = tt.swap_gate(axes=(1, 4))  # swap(1t, 0b)
        tt = tt.fuse_legs(axes=(0, (2, 4), (1, 3)))  # (e2 e3) (1b 0b) (1t 0t)
        Af = self.A.fuse_legs(axes=((1, 0), (2, 3), 4))  # (1t 0t) (2t 3t) p
        Abf = self.Ab.fuse_legs(axes=((1, 0), 4, (2, 3)))  # (1b 0b) p (2b 3b)
        tt = tt.tensordot(Af, axes=(2, 0))  # (e2 e3) (1b 0b) (2t 3t) p
        tt = tt.tensordot(Abf, axes=((1, 3), (0, 1)), conj=(0, 1))  # (e2 e3) (2t 3t) (2b 3b)
        tt = tt.unfuse_legs(axes=(1, 2))  # (e2 e3) 2t 3t 2b 3b
        tt = tt.swap_gate(axes=(1, 4))  # swap(2t, 3b)
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))  # (e2 e3) (2t 2b) (3t 3b)
        tt = tt.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3))  # e2 (2t 2b) e3 (3t 3b)
        return tt

    def append_a_br(self, tt):
        """ Append the A and Ab tensors of self to the bottom-right corner, tt. """

        tt = tt.fuse_legs(axes=((0, 3), 1, 2))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(2, 3))
        tt = tt.fuse_legs(axes=(0, (2, 4), (1, 3)))
        Af = self.A.fuse_legs(axes=((3, 2), (0, 1), 4))
        Abf = self.Ab.fuse_legs(axes=((3, 2), 4, (0, 1)))
        tt = tt.tensordot(Af, axes=(2, 0))
        tt = tt.tensordot(Abf, axes=((1, 3), (0, 1)), conj=(0, 1))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(2, 3))
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))
        tt = tt.unfuse_legs(axes=0)
        tt = tt.transpose(axes=(0, 2, 1, 3))
        return tt

    def _attach_01(self, tt):
        """
        Attach a tensor to the top left corner of the tensor network tt if rotation = 0
        and to the bottom left if rotation is 90.
  
        """
        if self._r == 0:
            return self.append_a_tl(tt)
        elif self._r == 90:
            return self.append_a_bl(tt)
    
    def _attach_23(self, tt):
        """
        Attach a tensor to the bottom right corner of the tensor network tt if rotation = 0
        and to the top right if rotation is 90.
        """
        if self._r == 0:
            return self.append_a_br(tt)
        elif self._r == 90:
            return self.append_a_tr(tt)

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
