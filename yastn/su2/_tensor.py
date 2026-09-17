"""Reduced SU(2) tensors using a left-associated Clebsch--Gordan tree."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import factorial, sqrt

import numpy as np

from ..sym import sym_SU2, sym_SU2xU1

__all__ = ['Leg', 'SU2U1Leg', 'SU2Tensor', 'SU2U1Tensor', 'tensordot']


def _spin(label):
    return label if isinstance(label, int) else label[0]


def _fact_half(two_n):
    """``(two_n / 2)!`` for a non-negative even integer ``two_n``."""
    if two_n < 0 or two_n % 2:
        return 0
    return factorial(two_n // 2)


@lru_cache(maxsize=None)
def clebsch_gordan(j1, m1, j2, m2, J, M):
    """Return ``<j1 m1, j2 m2 | J M>`` (Condon--Shortley convention).

    Every argument is twice its conventional angular-momentum value.  The
    implementation uses only integer factorials and has no SymPy dependency.
    """
    if M != m1 + m2 or J not in sym_SU2.fusion_outcomes(j1, j2) or abs(M) > J:
        return 0.0
    terms = (J + j1 - j2, J - j1 + j2, j1 + j2 - J)
    numerator = (J + 1) * _fact_half(terms[0]) * _fact_half(terms[1]) * _fact_half(terms[2])
    denominator = _fact_half(j1 + j2 + J + 2)
    numerator *= (_fact_half(J + M) * _fact_half(J - M) *
                  _fact_half(j1 - m1) * _fact_half(j1 + m1) *
                  _fact_half(j2 - m2) * _fact_half(j2 + m2))
    if not denominator:
        return 0.0
    prefactor = sqrt(numerator / denominator)
    total = 0.0
    # The six factorials in the Racah sum determine a small finite range.
    for k in range(0, (j1 + j2 - J) // 2 + 1):
        args = (2 * k, j1 + j2 - J - 2 * k, j1 - m1 - 2 * k,
                j2 + m2 - 2 * k, J - j2 + m1 + 2 * k,
                J - j1 - m2 + 2 * k)
        den = 1
        for arg in args:
            value = _fact_half(arg)
            if not value:
                den = 0
                break
            den *= value
        if den:
            total += (-1.0) ** k / den
    return prefactor * total


class Leg:
    """A direct sum of SU(2) irreps.

    ``t`` contains non-negative ``two_j`` labels.  ``D`` is the degeneracy of
    each irrep, not its full dimension; thus ``Leg(t=(1,), D=(3,))`` has dense
    dimension ``3 * (1 + 1) == 6``.
    """
    def __init__(self, t, D, s=1):
        self.s = int(s)
        if self.s not in (-1, 1):
            raise ValueError("SU2 Leg signature must be -1 or +1")
        self.t = tuple(sym_SU2.validate_irrep(j) for j in t)
        self.D = tuple(int(d) for d in D)
        if not self.t or len(self.t) != len(self.D) or any(d <= 0 for d in self.D):
            raise ValueError("SU2 Leg needs matching non-empty, positive t and D")
        if tuple(sorted(self.t)) != self.t or len(set(self.t)) != len(self.t):
            raise ValueError("SU2 Leg irreps must be unique and sorted")

    def conj(self):
        return Leg(self.t, self.D, -self.s)

    @property
    def dim(self):
        return sum(d * (j + 1) for j, d in zip(self.t, self.D))

    def __repr__(self):
        return f"SU2Leg(s={self.s}, t={self.t}, D={self.D})"


class SU2U1Leg:
    """A direct sum of ``(two_j, q)`` irreps of ``SU(2) x U(1)``."""
    def __init__(self, t, D, s=1):
        self.s = int(s)
        if self.s not in (-1, 1):
            raise ValueError("SU2xU1 Leg signature must be -1 or +1")
        self.t = tuple(sym_SU2xU1.validate_irrep(x) for x in t)
        self.D = tuple(int(d) for d in D)
        if not self.t or len(self.t) != len(self.D) or any(d <= 0 for d in self.D):
            raise ValueError("SU2xU1 Leg needs matching non-empty, positive t and D")
        if tuple(sorted(self.t)) != self.t or len(set(self.t)) != len(self.t):
            raise ValueError("SU2xU1 Leg irreps must be unique and sorted")

    def conj(self):
        # As in yastn.Leg, orientation is encoded in ``s``.  Keeping sector
        # labels here is essential: changing both ``s`` and q would reverse
        # the U(1) charge twice.  A charge-flipped basis, when needed, is a
        # distinct explicitly constructed leg.
        return SU2U1Leg(self.t, self.D, -self.s)

    @property
    def dim(self):
        return sum(d * (j + 1) for (j, _), d in zip(self.t, self.D))

    def __repr__(self):
        return f"SU2U1Leg(s={self.s}, t={self.t}, D={self.D})"


def _ms(j):
    return tuple(range(-j, j + 1, 2))


def _paths(js):
    """Allowed left-associated paths which fuse all ``js`` to a singlet."""
    if len(js) == 0:
        return ((),)
    if len(js) == 1:
        return ((),) if js[0] == 0 else ()
    paths = [()]
    current = [js[0]]
    for right in js[1:-1]:
        next_paths, next_current = [], []
        for path, left in zip(paths, current):
            for out in sym_SU2.fusion_outcomes(left, right):
                next_paths.append(path + (out,))
                next_current.append(out)
        paths, current = next_paths, next_current
    return tuple(path for path, left in zip(paths, current) if 0 in sym_SU2.fusion_outcomes(left, js[-1]))


@lru_cache(maxsize=2048)
def _intertwiner(js, path):
    """Normalized invariant tensor for the indicated irreps and fusion path."""
    if len(js) == 0:
        return np.array(1.0)
    if len(js) == 1:
        return np.array([1.0]) if js[0] == 0 else None
    if len(path) != len(js) - 2 or path not in _paths(js):
        raise ValueError(f"Invalid SU2 fusion path {path} for irreps {js}")
    out = np.zeros(tuple(j + 1 for j in js), dtype=float)
    for inds in product(*[range(j + 1) for j in js]):
        m = tuple(2 * i - j for i, j in zip(inds, js))
        left, ml, coefficient = js[0], m[0], 1.0
        for i, middle in enumerate(path):
            coefficient *= clebsch_gordan(left, ml, js[i + 1], m[i + 1], middle, ml + m[i + 1])
            left, ml = middle, ml + m[i + 1]
        coefficient *= clebsch_gordan(left, ml, js[-1], m[-1], 0, 0)
        out[inds] = coefficient
    return out


class SU2Tensor:
    """An SU(2)-invariant tensor represented by reduced (degeneracy) blocks.

    A block key is ``(irreps, path)``.  ``irreps`` chooses one sector on each
    leg; ``path`` lists intermediate ``two_j`` values of the left-associated
    fusion tree.  For a rank-three singlet the path has one value, e.g.
    ``((1, 1, 0), (1,))``.
    """
    def __init__(self, legs, blocks=None, dtype=None):
        self.legs = tuple(legs)
        if not all(isinstance(leg, Leg) for leg in self.legs):
            raise TypeError("legs must be yastn.su2.Leg instances")
        self.blocks = {}
        for key, value in (blocks or {}).items():
            irreps, path = tuple(key[0]), tuple(key[1])
            self._validate_key(irreps, path)
            shape = tuple(leg.D[leg.t.index(j)] for leg, j in zip(self.legs, irreps))
            value = np.asarray(value, dtype=dtype)
            # A scalar is the natural notation for a block whose every
            # degeneracy dimension equals one.
            if value.ndim == 0 and int(np.prod(shape, dtype=int)) == 1:
                value = value.reshape(shape)
            if value.shape != shape:
                raise ValueError(f"Block {key} has shape {value.shape}; expected {shape}")
            self.blocks[(irreps, path)] = value.copy()

    def _validate_key(self, irreps, path):
        if len(irreps) != len(self.legs) or any(j not in leg.t for j, leg in zip(irreps, self.legs)):
            raise ValueError("Block irreps do not match tensor legs")
        _intertwiner(tuple(_spin(x) for x in irreps), tuple(path))

    @property
    def shape(self):
        return tuple(leg.dim for leg in self.legs)

    @property
    def ndim(self):
        return len(self.legs)

    def to_dense(self):
        """Expand reduced blocks with Clebsch--Gordan coefficients."""
        dense = np.zeros(self.shape, dtype=np.result_type(*([b.dtype for b in self.blocks.values()] or [float])))
        offsets = [np.cumsum((0,) + tuple(d * (_spin(j) + 1) for j, d in zip(leg.t, leg.D))) for leg in self.legs]
        for (js, path), block in self.blocks.items():
            spins = tuple(_spin(x) for x in js)
            coefficient = _intertwiner(spins, path)
            expanded = block.reshape(tuple(x for d in block.shape for x in (d, 1)))
            expanded = expanded * coefficient.reshape(tuple(x for j in spins for x in (1, j + 1)))
            expanded = expanded.reshape(tuple(d * (j + 1) for d, j in zip(block.shape, spins)))
            slices = tuple(slice(offsets[a][leg.t.index(j)], offsets[a][leg.t.index(j) + 1])
                           for a, (leg, j) in enumerate(zip(self.legs, js)))
            dense[slices] += expanded
        return dense

    @classmethod
    def from_dense(cls, dense, legs, atol=1e-12):
        """Project a dense invariant tensor onto reduced SU(2) blocks.

        A non-invariant component raises ``ValueError`` rather than being
        silently discarded.
        """
        result = cls(legs)
        dense = np.asarray(dense)
        if dense.shape != result.shape:
            raise ValueError(f"Dense tensor has shape {dense.shape}; expected {result.shape}")
        offsets = [np.cumsum((0,) + tuple(d * (_spin(j) + 1) for j, d in zip(leg.t, leg.D))) for leg in result.legs]
        reconstructed = np.zeros_like(dense)
        for js in product(*[leg.t for leg in result.legs]):
            slices = tuple(slice(offsets[a][leg.t.index(j)], offsets[a][leg.t.index(j) + 1])
                           for a, (leg, j) in enumerate(zip(result.legs, js)))
            sector = dense[slices]
            Ds = tuple(leg.D[leg.t.index(j)] for leg, j in zip(result.legs, js))
            sector = sector.reshape(tuple(x for d, j in zip(Ds, js) for x in (d, _spin(j) + 1)))
            sector = sector.transpose(tuple(range(0, 2 * result.ndim, 2)) + tuple(range(1, 2 * result.ndim, 2)))
            spins = tuple(_spin(x) for x in js)
            for path in _paths(spins):
                coefficient = _intertwiner(spins, path)
                block = np.tensordot(sector, coefficient, axes=(tuple(range(result.ndim, 2 * result.ndim)), tuple(range(result.ndim))))
                if np.any(block):
                    result.blocks[(tuple(js), path)] = block
        reconstructed = result.to_dense()
        if not np.allclose(dense, reconstructed, atol=atol, rtol=0):
            raise ValueError("Dense data is not SU2-invariant for the supplied legs")
        return result

    def tensordot(self, other, axes):
        return tensordot(self, other, axes)

    def transpose(self, axes=None):
        """Permute legs while retaining reduced blocks and fusion channels."""
        axes = tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError('axes must be a permutation of tensor axes')
        legs = tuple(self.legs[x] for x in axes)
        return type(self).from_dense(self.to_dense().transpose(axes), legs)

    def conj(self):
        """Complex conjugate data and reverse all leg orientations."""
        legs = tuple(leg.conj() for leg in self.legs)
        # SU(2) dualisation is performed by the invariant pairing on every
        # leg.  Reusing the contraction metric keeps the Condon--Shortley
        # convention consistent with tensordot.
        data = self.to_dense().conj()
        for axis, leg in enumerate(self.legs):
            data = np.tensordot(_dual_metric(leg), data, axes=(1, axis))
            data = np.moveaxis(data, 0, axis)
        return type(self).from_dense(data, legs)

    def copy(self):
        return type(self)(self.legs, self.blocks)

    def norm(self):
        return float(np.linalg.norm(self.to_dense()))

    def __mul__(self, number):
        if not np.isscalar(number):
            return NotImplemented
        return type(self)(self.legs, {key: number * value for key, value in self.blocks.items()})

    __rmul__ = __mul__

    def __add__(self, other):
        if type(self) is not type(other) or self.legs != other.legs:
            raise ValueError('Can only add SU2 tensors with identical legs and symmetry')
        keys = self.blocks.keys() | other.blocks.keys()
        return type(self)(self.legs, {key: self.blocks.get(key, 0) + other.blocks.get(key, 0) for key in keys})

    def __sub__(self, other):
        return self + (-1) * other

    def to_dict(self):
        """Serialize in a new, versioned format; legacy Tensor files are unchanged."""
        return {'type': 'SU2Tensor', 'dict_ver': 1,
                'legs': [{'s': leg.s, 't': leg.t, 'D': leg.D} for leg in self.legs],
                'blocks': [{'irreps': js, 'path': path, 'data': data.tolist()}
                           for (js, path), data in self.blocks.items()]}

    @classmethod
    def from_dict(cls, data, config=None):
        if config is not None:
            raise ValueError("SU2Tensor carries no legacy Tensor config; do not override it")
        if data.get('type') != 'SU2Tensor' or data.get('dict_ver') != 1:
            raise ValueError("Not an SU2Tensor dictionary (version 1 expected)")
        legs = tuple(Leg(**leg) for leg in data['legs'])
        blocks = {(tuple(item['irreps']), tuple(item['path'])): np.asarray(item['data'])
                  for item in data['blocks']}
        return cls(legs, blocks)


class SU2U1Tensor(SU2Tensor):
    """Reduced invariant tensor for the product symmetry ``SU(2) x U(1)``."""
    symmetry = sym_SU2xU1

    def __init__(self, legs, blocks=None, dtype=None):
        self.legs = tuple(legs)
        if not all(isinstance(leg, SU2U1Leg) for leg in self.legs):
            raise TypeError('legs must be yastn.su2.SU2U1Leg instances')
        self.blocks = {}
        for key, value in (blocks or {}).items():
            irreps, path = tuple(key[0]), tuple(key[1])
            self._validate_key(irreps, path)
            shape = tuple(leg.D[leg.t.index(j)] for leg, j in zip(self.legs, irreps))
            value = np.asarray(value, dtype=dtype)
            if value.ndim == 0 and int(np.prod(shape, dtype=int)) == 1:
                value = value.reshape(shape)
            if value.shape != shape:
                raise ValueError(f"Block {key} has shape {value.shape}; expected {shape}")
            self.blocks[(irreps, path)] = value.copy()

    def _validate_key(self, irreps, path):
        if len(irreps) != len(self.legs) or any(j not in leg.t for j, leg in zip(irreps, self.legs)):
            raise ValueError('Block irreps do not match tensor legs')
        if sum(leg.s * irrep[1] for leg, irrep in zip(self.legs, irreps)) != 0:
            raise ValueError('SU2xU1 block violates the U1 selection rule')
        _intertwiner(tuple(_spin(x) for x in irreps), tuple(path))

    @classmethod
    def from_dict(cls, data, config=None):
        if config is not None:
            raise ValueError('SU2U1Tensor carries no legacy Tensor config; do not override it')
        if data.get('type') != 'SU2U1Tensor' or data.get('dict_ver') != 1:
            raise ValueError('Not an SU2U1Tensor dictionary (version 1 expected)')
        legs = tuple(SU2U1Leg(**leg) for leg in data['legs'])
        blocks = {(tuple(tuple(x) for x in item['irreps']), tuple(item['path'])): np.asarray(item['data'])
                  for item in data['blocks']}
        return cls(legs, blocks)

    def to_dict(self):
        return {'type': 'SU2U1Tensor', 'dict_ver': 1,
                'legs': [{'s': leg.s, 't': leg.t, 'D': leg.D} for leg in self.legs],
                'blocks': [{'irreps': js, 'path': path, 'data': data.tolist()}
                           for (js, path), data in self.blocks.items()]}


def tensordot(a, b, axes):
    """Contract two invariant SU(2) tensors and retain their reduced form."""
    if type(a) is not type(b) or not isinstance(a, SU2Tensor):
        raise TypeError("tensordot requires two reduced tensors of the same symmetry")
    axa, axb = axes
    axa, axb = (axa,) if isinstance(axa, int) else tuple(axa), (axb,) if isinstance(axb, int) else tuple(axb)
    if len(axa) != len(axb):
        raise ValueError("Contracted axis lists must have equal length")
    for x, y in zip(axa, axb):
        compatible_t = a.legs[x].t
        if compatible_t != b.legs[y].t or a.legs[x].D != b.legs[y].D or a.legs[x].s == b.legs[y].s:
            raise ValueError("Contracted SU2 legs must be conjugate and have identical sectors")
    # Contracting a dual SU(2) index uses the invariant epsilon metric rather
    # than a component-wise Kronecker delta.  This is what preserves the
    # reduced/intertwiner form after a contraction of half-integer irreps.
    bdense = b.to_dense()
    for y in axb:
        metric = _dual_metric(b.legs[y])
        bdense = np.tensordot(metric, bdense, axes=(1, y))
        bdense = np.moveaxis(bdense, 0, y)
    legs = tuple(leg for i, leg in enumerate(a.legs) if i not in axa) + tuple(leg for i, leg in enumerate(b.legs) if i not in axb)
    return type(a).from_dense(np.tensordot(a.to_dense(), bdense, axes=(axa, axb)), legs)


def _dual_metric(leg):
    """Block-diagonal SU(2)-invariant pairing for a leg and its dual."""
    metric = np.zeros((leg.dim, leg.dim), dtype=float)
    offset = 0
    for label, D in zip(leg.t, leg.D):
        j = _spin(label)
        eps = np.zeros((j + 1, j + 1), dtype=float)
        for index, m in enumerate(_ms(j)):
            eps[index, j - index] = (-1.0) ** ((j - m) // 2)
        metric[offset:offset + D * (j + 1), offset:offset + D * (j + 1)] = np.kron(np.eye(D), eps)
        offset += D * (j + 1)
    return metric
    symmetry = sym_SU2
