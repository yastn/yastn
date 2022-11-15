""" Class that treats a cell of a double-layer peps as a single tensor."""


class DoublePepsTensor:
    def __init__(self, top, bottom):
        self.A = top
        self.Ab = bottom



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
        phi = DoublePepsTensor()
        phi.A = self.A.clone()
        phi.Ab = self.Ab.clone()
        return phi

    def copy(self):
        r"""
        Makes a copy of DoublePepsTensor by :meth:`copying<yast.Tensor.copy>` all :class:`yast.Tensor<yast.Tensor>`'s
        into a new and independent :class:`yamps.MpsMpo`.

        .. warning::
            this operation does not preserve autograd on the returned :code:`yamps.MpsMpo`.

        .. note::
            Use when retaining "old" DoublePepsTensor is necessary. 

        Returns
        -------
        yast.tn.peps.DoublePepsTensor
            a copy of :code:`self`
        """
        phi = DoublePepsTensor()
        phi.A = self.A.copy()
        phi.Ab = self.Ab.copy()
        return phi

         
    def append_a_bl(self, tt):
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
