from __future__ import annotations

from ._doublePepsTensor import DoublePepsTensor
from ._gates_auxiliary import apply_gate_onsite
from ...tensor import Tensor, YastnError, tensordot


class DoublePepsTensorCharged(DoublePepsTensor):
    """
    Experimental charged-sector variant of ``DoublePepsTensor``.

    It reuses the standard double-layer bookkeeping but evaluates contractions
    through the fully fused double-layer tensor. This is slower than the
    production fast path, but it is much safer for experimental charged-sector
    work because it avoids delicate fermionic sign logic in the specialized
    auxiliary contractions.
    """

    def apply_gate_on_ket(self, op, dirn):
        ket = apply_gate_onsite(self.ket, op, dirn=dirn)
        return type(self)(bra=self.bra, ket=ket, transpose=self._t, op=self.op, swaps=self.swaps)

    def transpose(self, axes):
        axes = tuple(self._t[ax] for ax in axes)
        return type(self)(bra=self.bra, ket=self.ket, transpose=axes, op=self.op, swaps=self.swaps)

    def conj(self):
        op_conj = self.op.conj() if self.op is not None else None
        return type(self)(
            bra=self.bra.conj(),
            ket=self.ket.conj(),
            transpose=self._t,
            op=op_conj,
            swaps=self.swaps,
        )

    def clone(self):
        op_clone = self.op.clone() if self.op is not None else None
        return type(self)(
            bra=self.bra.clone(),
            ket=self.ket.clone(),
            transpose=self._t,
            op=op_clone,
            swaps=self.swaps,
        )

    def copy(self):
        op_copy = self.op.copy() if self.op is not None else None
        return type(self)(
            bra=self.bra.copy(),
            ket=self.ket.copy(),
            transpose=self._t,
            op=op_copy,
            swaps=self.swaps,
        )

    @classmethod
    def from_dict(cls, d, config=None):
        if cls.__name__ != d["type"]:
            raise YastnError(f"{cls.__name__} does not match d['type'] == {d['type']}")
        bra = Tensor.from_dict(d=d["bra"], config=config)
        ket = Tensor.from_dict(d=d["ket"], config=config)
        op = Tensor.from_dict(d=d["op"], config=config) if "op" in d else None
        return cls(bra=bra, ket=ket, transpose=d["transpose"], op=op, swaps=d["swaps"])

    def to_dict(self, level=2):
        d = super().to_dict(level=level)
        d["type"] = type(self).__name__
        return d

    def tensordot(self, b, axes, reverse=False):
        fused = self.fuse_layers()
        if reverse:
            return tensordot(b, fused, axes=axes)
        return tensordot(fused, b, axes=axes)

    def print_properties(self, file=None):
        print("DoublePepsTensorCharged", file=file)
        print("shape   :", self.get_shape(), file=file)
        st = {i: leg.history() for i, leg in enumerate(self.get_legs())}
        print("legs fusions :", st, "\n", file=file)
