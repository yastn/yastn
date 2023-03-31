from yast import tensordot, leg_outer_product

_rotations = {0: (0, 1, 2, 3), 90: (1, 2, 3, 0)}

class DoublePepsTensor:

    r"""Class that treats a cell of a double-layer peps as a single tensor.

    Parameters
    ----------
    top : yast.Tensor
        The top tensor of the cell.
    btm : yast.Tensor
        The bottom tensor of the cell.
    rotation : int, optional
        The rotation angle of the cell, default is 0.

    Attributes
    ----------
    A : yast.Tensor
        The top tensor of the cell.
    Ab : yast.Tensor
        The bottom tensor of the cell.
    _r : int
        The rotation angle of the cell.

    Methods
    -------
    ndim()
        Get the number of dimensions of the tensor.
    get_shape()
        Get the shape of the tensor.
    get_legs()
        Get the tensor legs of the tensor.
    clone()
        Make a clone of the tensor with autograd tracking preserved.
    copy()
        Make a copy of the tensor without autograd tracking preserved.
    append_a_bl
        Append a tensor to the bottom-left of the cell.
    append_a_tr
        Append a tensor to the top-right of the cell.
    """

    def __init__(self, top, btm, rotation=0):
        self.A = top
        self.Ab = btm
        self._r = rotation

    @property
    def ndim(self):
        return 4

    def get_shape(self, axis=None):
        """ Returns the shape of the DoublePepsTensor along the specified axes """

        if axis is None:
            axis = tuple(range(4))
        sA = self.A.get_shape(axis=axis)
        sB = self.Ab.get_shape(axis=axis)
        if isinstance(axis, int):
            return sA * sB 
        return tuple(x * y for x, y in zip(sA, sB))

    def get_legs(self, axis=None):
        """ Returns the legs of the DoublePepsTensor along the specified axes. """

        if axis is None:
            axis = tuple(range(4))
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        rot = _rotations[self._r]
        axes = tuple(rot[ax] for ax in axes)
        lts = self.A.get_legs(axis=axes)
        lbs = self.A.get_legs(axis=axes)
        if hasattr(axis, '__iter__'):
            lts, lbs = (lts,), (lbs,)
        legs = []
        for lt, lb in zip(lts, lbs):
            legs.append(leg_outer_product(lt, lb.conj()))
        return tuple(legs) if hasattr(axis, '__iter__') else legs.pop()


    def clone(self):
        r"""
        Makes a clone of Double Peps Tensor by :meth:`cloning<yast.Tensor.clone>`
        all :class:`yast.Tensor<yast.Tensor>`'s into a new and independent :class:`peps.DoublePepsTensor`.

        .. note::
            Cloning preserves autograd tracking on all tensors.

        Returns
        -------
        yast.tn.peps.DoublePepsTensor
            a clone of :code:`self`
        """
        return DoublePepsTensor(self.A.clone(), self.Ab.clone(), rotation=self._r)

    def copy(self):
        r"""
        Makes a copy of DoublePepsTensor by :meth:`copying<yast.Tensor.copy>` all :class:`yast.Tensor<yast.Tensor>`'s
        into a new and independent :class:`yast.tn.mps.MpsMpo`.

        .. warning::
            this operation does not preserve autograd on the returned :code:`yast.tn.mps.MpsMpo`.

        .. note::
            Use when retaining "old" DoublePepsTensor is necessary. 

        Returns
        -------
        yast.tn.peps.DoublePepsTensor
            a copy of :code:`self`
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

        tt = tt.fuse_legs(axes=((0, 3), 1, 2))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(1, 4))
        tt = tt.fuse_legs(axes=(0, (2, 4), (1, 3)))
        Af = self.A.fuse_legs(axes=((1, 0), (2, 3), 4))
        Abf = self.Ab.fuse_legs(axes=((1, 0), 4, (2, 3)))
        tt = tt.tensordot(Af, axes=(2, 0))
        tt = tt.tensordot(Abf, axes=((1, 3), (0, 1)), conj=(0, 1))
        tt = tt.unfuse_legs(axes=(1, 2))
        tt = tt.swap_gate(axes=(1, 4))
        tt = tt.fuse_legs(axes=(0, (1, 3), (2, 4)))
        tt = tt.unfuse_legs(axes=0).transpose(axes=(0, 2, 1, 3))
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
        Attach a tensor to the top left corner of the tensor network if rotation = 0
        and to the bottom left if rotation is 90.
        Parameters
        ----------
            tt: tensor to attach
        Returns
        -------
            tensor network with tensor tt attached
        """
        if self._r == 0:
            return self.append_a_tl(tt)
        elif self._r == 90:
            return self.append_a_bl(tt)
    
    def _attach_23(self, tt):
        """
        Attach a tensor to the bottom right corner of the tensor network if rotation = 0
        and to the top right if rotation is 90.
        Parameters
        ----------
            tt: tensor to attach
        Returns
        -------
            tensor network with tensor tt attached
        """
        if self._r == 0:
            return self.append_a_br(tt)
        elif self._r == 90:
            return self.append_a_tr(tt)

    def fPEPS_fuse_layers(self):

        """
          Fuse the top and bottom layers of a PEPS tensor network.
        Returns
        -------
          A new tensor network obtained by fusing the top and bottom layers.
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
