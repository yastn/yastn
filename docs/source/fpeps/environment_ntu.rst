Environment NTU
===============

Neighborhood tensor update environments
---------------------------------------

.. autoclass:: yastn.tn.fpeps.EnvNTU
    :members: bond_metric

Streaming the metric
--------------------

To avoid storing the complete bond metric, set ``metric_chunk_size``::

    env = fpeps.EnvNTU(psi, which='NN', metric_chunk_size=16)
    infos = fpeps.evolution_step_(
        env, gates, opts_svd={'D_total': 12},
        fix_metric=None, initialization='EAT_SVD', max_iter=20)

The first open QR leg is partitioned into orthogonal subspaces. If ``P_i``
embeds subspace ``i``, then ``sum_i P_i @ P_i.H = I``. Both layers are
partitioned independently, and every block ``g_ij`` contributes to metric
actions and ALS normal matrices. This retains all cross terms, analogous to
expanding ``g(G, G*)`` for a sum of gates. The implementation partitions the
common QR basis after gate application; it does not apply and truncate the
gate terms separately, or add metrics computed in different QR bases.
No explicit decomposition of the input gate is required.

``metric_chunk_size`` counts multiplicities, so an SU(2) multiplet is never
split. Blocks are contracted on demand and discarded, including during
truncation-error evaluation. The full metric is not assembled. The two
smaller ALS normal matrices are still stored and diagonalized. Smaller
chunks reduce metric-block memory at the cost of repeated contractions;
they do not impose a bound on total memory, which also includes the PEPS,
QR factors, environment intermediates and ALS matrices.
If the chunk size covers the entire first QR leg, there is just one block
and no metric-storage saving.

All NTU environment sizes support this mode. NN gates and multi-site MPO
paths, including NNN gates through an intermediate site, use the same
streaming implementation at each truncated bond. ``which='NNN'`` specifies
the environment size and is independent of the range of the gate.
The existing SU(2) ``Ladder2`` signature incompatibility also affects
streaming; use one of the six standard NTU environments for SU(2).
For purification gates, use ``default_fusion='hard'``: the existing
``match_ancilla`` path does not handle meta-fused physical legs.

Streaming requires ``fix_metric=None``. Supported initializations are
``'SVD'``, ``'SVD_ONLY'``, ``'EAT'``, ``'EAT_SVD'`` and ``'SVD_EAT'``.
Full-metric eigenvalue correction and ZMT initialization
are unavailable in this mode; incompatible options raise an error before
the PEPS is modified. Hermiticity is enforced blockwise, and
``nonhermitian_part`` is still measured. ``min_eigenvalue`` and
``wrong_eigenvalues`` remain ``None``. With the default
``metric_chunk_size=None``, the original full-metric behavior is unchanged.

Streaming EAT computes the leading product approximation using alternating
power iteration on the reshuffled metric. Each factor update contracts and
discards the same metric blocks; neither the full metric nor its reshuffled
matrix is assembled. The identity initializes the right factor, and all
tensor contractions remain on the configured backend/device. The resulting
factors feed the existing EAT initialization and subsequent ALS optimization.
``eat_metric_error`` uses the singular value and the metric norm accumulated
from the blocks. Factors are reused if the bond is truncated in stages.

The optional ``opts_eat`` argument controls this iteration independently of
the ALS ``max_iter`` and ``tol_iter`` arguments::

    env = fpeps.EnvNTU(
        psi, metric_chunk_size=16,
        opts_eat={'tol': 1e-10, 'max_iter': 100})

These are the defaults. ``tol`` bounds the relative singular-pair residual;
it does not simply test for a small change in singular value. Slow convergence
is possible when the two largest singular values are close. Failure to reach
the tolerance raises ``YastnError`` with the final residual, without falling
back to a full metric. As with other failures during evolution, the gate may
already have been applied when this error is raised.


Approximate cluster update
--------------------------

.. autoclass:: yastn.tn.fpeps.EnvApproximate
    :members: bond_metric
