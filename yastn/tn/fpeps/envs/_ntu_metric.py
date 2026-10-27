"""Stream NTU metric blocks in a common QR basis."""
from itertools import combinations_with_replacement
from math import isfinite

from ....initialize import eye
from ....tensor import tensordot, YastnError


class NTUMetric:
    """Exact metric action without storing the full four-leg metric.

    Partition the first open QR leg into orthogonal subspaces. Both layers
    are partitioned independently, so all off-diagonal cross terms survive.
    Blocks are recomputed on demand, trading contractions for memory.
    """

    def __init__(self, env, Q0, Q1, s0, s1, dirn, chunk_size):
        axis = 3 if dirn in ('h', 'lr') else 2
        leg = Q0.get_legs(axis)
        identity = eye(Q0.config, legs=(leg.conj(), leg),
                       isdiag=False, device=Q0.device)
        mask = eye(Q0.config, legs=leg, device=Q0.device)
        self.projectors = []
        # Diagonal storage counts multiplicities, never splits an SU(2) irrep.
        for start in range(0, mask.size, chunk_size):
            select = mask * 0
            select._data[start:start + chunk_size] = 1
            self.projectors.append(select.apply_mask(identity, axes=1))
        if not self.projectors:
            raise YastnError('Cannot partition an empty NTU QR leg.')
        self.env, self.Q0, self.Q1 = env, Q0, Q1
        self.s0, self.s1, self.dirn, self.axis = s0, s1, dirn, axis
        self.nonhermitian_part = None
        self.metric_norm = None
        self._eat_factors = None

    def _q(self, projector):
        q = tensordot(self.Q0, projector, axes=(self.axis, 0))
        order = list(range(4))
        order.insert(self.axis, 4)
        return q.transpose(axes=order)

    def _raw_block(self, row, col):
        return self.env._bond_metric(self._q(col), self.Q1,
                                     self.s0, self.s1, self.dirn,
                                     Q0_ket=self._q(row)).g

    def blocks(self):
        """Yield hermitianized blocks, including every cross term."""
        measure = self.nonhermitian_part is None
        norm2, noise2 = 0., 0.
        for i, j in combinations_with_replacement(range(len(self.projectors)), 2):
            row, col = self.projectors[i], self.projectors[j]
            raw = self._raw_block(row, col)
            adjoint = raw.H if i == j else self._raw_block(col, row).H
            block = (raw + adjoint) / 2
            if measure:
                weight = 1 if i == j else 2
                norm2 += weight * block.norm().item() ** 2
                noise2 += weight * ((raw - adjoint).norm().item() / 2) ** 2
            del raw, adjoint
            yield row, col, block
            if i != j:
                yield col, row, block.H
            # Do not retain the previous block while contracting the next one.
            del block
        if measure:
            if norm2 == 0:
                raise YastnError('NTU metric has zero norm.')
            self.nonhermitian_part = (noise2 / norm2) ** 0.5
            self.metric_norm = norm2 ** 0.5

    def _eat_action(self, matrix, side):
        """Contract the reshuffled metric with one EAT factor.

        G has legs (left_row, right_row, left_col, right_col).
        EAT reshuffles it into ((0, 2), (3, 1)). The factors have
        legs (0, 2) and (3, 1), respectively; both updates conjugate
        the supplied factor, as in alternating rank-one SVD updates.
        """
        result = None
        for row, col, block in self.blocks():
            # Materialize the adjoint block's pending permutation before
            # exposing both non-Abelian fusion trees.
            G = block.consume_transpose()
            G = G.unfuse_legs(axes=(0, 1))
            if side == 0:
                value = tensordot(G, matrix, axes=((1, 3), (1, 0)), conj=(0, 1))
                value = row @ value @ col.H
            else:
                selected = row.H @ matrix @ col
                value = tensordot(selected, G, axes=((0, 1), (0, 2)), conj=(1, 0)).T
                del selected
            result = value if result is None else result + value
            del G, block, value
        return result

    def eat_factors(self):
        """Leading product approximation via streamed power iteration.

        Start with the identity, a positive definite factor. For the positive
        NTU metric, the alternating maps preserve positivity and access a
        leading positive singular pair in the invariant sector. Stop on the
        relative singular-pair residual, not on a change of singular value.
        Only metric blocks and the two matrix factors are stored.
        """
        if self._eat_factors is not None:
            return self._eat_factors
        axis = 1 if self.dirn in ('h', 'lr') else 0
        right = eye(self.Q1.config, legs=self.Q1.get_legs(axis),
                    isdiag=False, device=self.Q1.device)
        right = right / right.norm()
        tol, max_iter = self.env.opts_eat['tol'], self.env.opts_eat['max_iter']
        for iteration in range(1, max_iter + 1):
            left = self._eat_action(right, side=0)
            sigma = left.norm().item()
            if not isfinite(sigma) or sigma <= 0:
                raise YastnError('Streamed EAT encountered a zero or non-finite singular value.')
            left = left / sigma
            updated = self._eat_action(left, side=1)
            residual = (updated - sigma * right).norm().item() / sigma
            if residual <= tol:
                fid = sigma / self.metric_norm
                error = max(0., 1 - fid ** 2) ** 0.5
                self._eat_factors = left, right, error
                return self._eat_factors
            norm = updated.norm().item()
            if not isfinite(norm) or norm <= 0:
                raise YastnError('Streamed EAT encountered a zero or non-finite factor.')
            right = updated / norm
        raise YastnError(f'Streamed EAT did not converge after {iteration} iterations: '
                         f'relative residual={residual:.3e}, tol={tol:.3e}. '
                         'Increase opts_eat["max_iter"] or relax opts_eat["tol"].')

    def __matmul__(self, vector):
        matrix = vector.unfuse_legs(axes=0)
        result = None
        for row, col, block in self.blocks():
            selected = (col.H @ matrix).fuse_legs(axes=[(0, 1)])
            value = row @ (block @ selected).unfuse_legs(axes=0)
            result = value if result is None else result + value
            del block, selected, value
        return result.fuse_legs(axes=[(0, 1)])

    def normal_matrix(self, M, side):
        """Accumulate the smaller ALS normal matrix before solving it."""
        result = None
        for row, col, block in self.blocks():
            gf = block.unfuse_legs(axes=0)
            if side == 0:
                g = tensordot(M.conj(), gf, axes=(1, 1))
                g = g.fuse_legs(axes=((1, 0), 2))
                g = g.unfuse_legs(axes=1)
                g = tensordot(g, M, axes=(2, 1))
                g = g.unfuse_legs(axes=0)
                # g: row_left, row_bond, col_left, col_bond
                g = tensordot(row, g, axes=(1, 0))
                g = tensordot(g, col.conj(), axes=(2, 1))
                g = g.fuse_legs(axes=((0, 1), (3, 2)))
            else:
                mr, mc = row.H @ M, col.H @ M
                g = tensordot(mr.conj(), gf, axes=(0, 0))
                g = g.fuse_legs(axes=((0, 1), 2))
                g = g.unfuse_legs(axes=1)
                g = tensordot(g, mc, axes=(1, 0))
                g = g.fuse_legs(axes=(0, (2, 1)))
                del mr, mc
            result = g if result is None else result + g
            del block, gf, g
        return result
