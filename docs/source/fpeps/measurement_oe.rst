Exact n-site measurement with opt_einsum
=========================================

:meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_exact` contracts the CTM environment
around the rectangular window spanned by the requested sites exactly, row by
row.  :meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_exact_oe` computes the same
number by handing the whole window, corners, edges, ket, bra and operators,
to `opt_einsum` as one tensor network.  The contraction path is optimized,
individual bonds can be *unrolled* (sliced) to bound peak memory, and the norm
and the numerator can be evaluated separately so that many operators on the same
window share one norm contraction.

.. automethod:: yastn.tn.fpeps.EnvCTM.measure_nsite_exact_oe
.. automethod:: yastn.tn.fpeps.EnvCTM.measure_nsite_norm_exact_oe
.. automethod:: yastn.tn.fpeps.EnvCTM.measure_nsite_numerator_exact_oe


.. _oe-bond-labels:

Network layout, bond labels and unrolling
-----------------------------------------

The contraction builds a tensor network over the ``Nx`` x ``Ny`` window
enclosing the requested ``sites``: the four CTM corners ``TL, TR, BL, BR``,
the edge tensors ``T[j], B[j], L[i], R[i]``, and one double-layer site
``*`` per window position.  Every edge of that network carries a tuple
label; the ``unroll`` dict refers to those labels.  Coordinates ``i`` (row,
``0 ... Nx-1``) and ``j`` (column, ``0 ... Ny-1``) are window-local, not
absolute lattice positions, and follow the ``Site(x, y) = (row, col)``
convention::

               j=-1            j=0             j=1          j=Ny-1     j=Ny
                :              :               :             :          :
        i=-1   TL --h,-1,-1-- T[0] --h,-1,0-- T[1] -- ... -- h,-1,Ny-1 -- TR
                |              |               |             |          |
             v,0,-1         v,0,0           v,0,1        v,0,Ny-1     v,0,Ny
                |              |               |             |          |
        i=0    L[0]-h,0,-1-----*---h,0,0-------*--- ... --h,0,Ny-1-----R[0]
                |              |               |             |          |
             v,1,-1         v,1,0           v,1,1        v,1,Ny-1     v,1,Ny
                |              |               |             |          |
        i=1    L[1]-h,1,-1-----*---h,1,0-------*--- ... --h,1,Ny-1-----R[1]
                :              :               :             :          :
                |              |               |             |          |
             v,Nx,-1        v,Nx,0          v,Nx,1       v,Nx,Ny-1    v,Nx,Ny
                |              |               |             |          |
        i=Nx   BL --h,Nx,-1-- B[0] --h,Nx,0-- B[1] -- ... -- h,Nx,Ny-1 -- BR

* **Horizontal bonds** ``('h', i, j)`` run left to right between columns
  ``j`` and ``j+1`` at row ``i``.  Rows ``i = -1`` and ``i = Nx`` are the
  boundary rows (chi bonds between edge tensors and corners); ``i = 0 ...
  Nx-1`` are PEPS rows; ``j = -1`` is the left-boundary column and
  ``j = Ny-1`` the right-boundary column for the chi bonds attached to the
  side edges.
* **Vertical bonds** ``('v', i, j)`` run top to bottom in column ``j``
  between rows ``i-1`` and ``i``: ``i = 0`` connects the top row to the first
  PEPS row, ``i = Nx`` the last PEPS row to the bottom row, ``i = 1 ...
  Nx-1`` are interior; ``j = -1`` and ``j = Ny`` are the left and right
  boundary columns.
* **Ket / bra split.**  For ``DoublePepsTensor`` PEPS every PEPS-row bond,
  i.e. ``('h', i, j)`` with ``0 <= i < Nx`` and *all* ``('v', i, j)``,
  carries two labels, ``(*, 'k')`` for the ket layer and ``(*, 'b')`` for the
  bra layer.  An un-qualified label in ``unroll`` is expanded into both; a
  layer-qualified label like ``('v', 1, 1, 'k')`` slices only the ket side.
  Boundary (chi) bonds are single-label.
* **Operator bonds** ``('opb', k)`` label the bond between MPO tensors
  ``k-1`` and ``k`` (next section); they exist only in the numerator network
  and are ignored by the norm.

Examples for a 2 x 3 window (``Nx = 2``, ``Ny = 3``)::

    # unroll the horizontal bond between columns 0 and 1 at the first PEPS
    # row, one index at a time (an int is a uniform slice size):
    unroll = {('h', 0, 0): 1}

    # unroll the vertical bond in column 1 between rows 0 and 1:
    unroll = {('v', 1, 1): 1}

    # several bonds at once; a left-boundary (chi) bond:
    unroll = {('h', 0, 0): 1, ('v', 1, 1): 1}
    unroll = {('v', 0, -1): 1}

    # one charge sector at a time:
    from yastn.tensor.oe_blocksparse import make_sliced_legs
    unroll = {('h', 0, 0): make_sliced_legs(leg)}

With the default ``separate_layers=True`` the ket, the operator and the bra
of every site stay separate network tensors (:func:`_build_ketbra_separate`);
with ``separate_layers=False`` each site's ket and bra are pre-contracted
through the operator into one 8-leg tensor (:func:`_build_ketbra_contracted`).
The latter is only possible for plain two-leg operators, whose Jordan-Wigner
strings carry a fixed charge and can be applied to the site tensors before
the contraction; see the next section.


Strings of plain charged operators
----------------------------------

A plain operator of non-zero charge :math:`q` (a single :math:`c` or
:math:`c^\dagger`) is the MPO case with a bond of dimension one: its
virtual leg carries the fixed charge :math:`q` and has to be routed to a
common reference point, the top-left corner of the window, so that the
Jordan-Wigner strings of all operators in the product meet there and every
crossing between them is accounted for.  The measurement takes the product in
the lattice's fermionic order, :math:`O_{s_1} O_{s_2} \cdots O_{s_n}` with
:math:`s_1 < s_2 < \cdots` (``sign_canonical_order`` supplies the sign of
bringing the listed order into this one).  The rightmost operator acts first,
and the string of :math:`O_s` runs over the sites *earlier* than :math:`s`,
whose operators have not acted yet.  Every string therefore sees the
occupation of the state before any operator, i.e. the bare ket physical leg.

Sites earlier than :math:`s = (x, y)` in the fermionic order are those above
it in column :math:`y` and all sites in the columns to its left.  The string
of :math:`O_s` is routed accordingly (:func:`_string_path`): up its own
column to the top row of the window, then left along the top row to the
corner.  Along the way it crosses the following legs of the double-layer site
tensors, named ``k`` (ket) or ``b`` (bra) plus the leg index
``0 = top, 1 = left, 2 = bottom, 3 = right, 4 = physical``::

    corner                         top row                       column y
    (minx, miny)     (minx, y1), miny < y1 < y      (minx, y)     (x1, y), minx < x1 < x      (x, y)

      k2, k4   <---   b0, k2, k4   <---   b0   <---  b3, k4  <---  k1, b3, k4  <---  k1

``k4`` is the ket's own physical leg, before the operator sitting on that
site acts.  Only the physical legs on the top row and in the site's own
column are crossed explicitly; the parity of the sites lower in the left
columns enters through the vertical ket and bra legs ``k2``, ``b0`` that the
line crosses on the top row, since the site tensors below the line are
neutral and their parity flows through those legs.

Each crossing is a swap gate with the fixed charge :math:`q`,
:math:`(-1)^{p(q)\,p(\mathrm{leg})}` block by block
(:meth:`yastn.tn.fpeps.DoublePepsTensor.add_charge_swaps_`).  Because the
charge is fixed, no extra network leg is needed: the gates are multiplied
into the ket and bra tensors of the crossed sites before the contraction, in
both builders.  Strings of several operators may overlap; on a shared leg the
charges add, so two strings of opposite charge cancel there.  This is the
same path :meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_exact` uses.


Operators as MPO tensors
------------------------

Building the MPO
^^^^^^^^^^^^^^^^

A sum of operator products on one set of sites,

.. math::

    O = \sum_t c_t \, o^{(t)}_{s_0} \, o^{(t)}_{s_1} \cdots o^{(t)}_{s_{L-1}},

can be measured in a single contraction instead of one contraction per term.
Build an MPO from the terms and pass it in place of the plain operators::

    mpo, bond_dims = fpeps.mpo_from_products(terms, tol=1e-12)
    value = env.measure_nsite_exact_oe(*mpo, sites=sites)

``terms`` is a list of ``(coeff, ops)`` pairs with one two-leg operator per
site; the leg order of the returned tensors is given in
:func:`yastn.tn.fpeps.mpo_from_products` below.

What the measurement requires
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* An operator of rank greater than two is read as an MPO tensor.  Its number
  of bond legs must match its position in the list: one for the two ends,
  two in the middle.
* Either every site carries an MPO tensor or every site carries a plain
  two-leg operator; the two kinds are never mixed in one call, and each site
  appears exactly once.
* ``sites`` must be listed in the lattice's fermionic order
  (:meth:`yastn.tn.fpeps.SquareLattice.f_ordered`); otherwise the call
  raises.  The operators inside every term handed to ``mpo_from_products``
  follow the same order.
* For MPO input the measurement applies no reordering sign of its own.  The
  commutation sign of each term must already sit in its coefficient.

To evaluate a term written in another order, permute it first::

    sign, perm = fpeps.canonical_order(term_ops, term_sites, env.f_ordered)
    term_sites = [term_sites[p] for p in perm]
    term_ops = [term_ops[p] for p in perm]
    coeff = sign * coeff

:func:`yastn.tn.fpeps.canonical_order` returns the permutation that sorts the
sites into fermionic order and the sign of commuting the operators along with it.

Operator bonds and unrolling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bond between MPO tensors ``k-1`` and ``k`` is labelled ``('opb', k)``.
It is unrolled like any other bond, e.g. ``unroll={('opb', 1): 2}``, or one
charge sector at a time with
:func:`yastn.tensor.oe_blocksparse.make_sliced_legs`.  The norm network
contains no operator bonds, so these entries are dropped when the norm is
contracted.  MPO tensors need ``separate_layers=True``, the default; a call
with ``separate_layers=False`` is switched over with a warning.

Example
^^^^^^^

Hopping plus density-density interaction on one bond of a fermionic PEPS with
CTM environment ``env``::

    import yastn
    import yastn.tn.fpeps as fpeps

    ops = yastn.operators.SpinlessFermions(sym='U1')
    c, cp, n = ops.c(), ops.cp(), ops.n()
    sites = [(0, 0), (0, 1)]                        # in fermionic order
    terms = [(-1.0, [cp, c]), (-1.0, [c, cp]), (0.5, [n, n])]
    mpo, bond_dims = fpeps.mpo_from_products(terms)   # bond_dims == [3]

    norm = env.measure_nsite_norm_exact_oe(sites=sites)
    num = env.measure_nsite_numerator_exact_oe(*mpo, sites=sites, unroll={('opb', 1): 1})
    value = num / norm                              # == sum of the three plain measurements

.. autofunction:: yastn.tn.fpeps.mpo_from_products
.. autofunction:: yastn.tn.fpeps.sum_of_products
.. autofunction:: yastn.tn.fpeps.canonical_order

Swap gates of MPO
^^^^^^^^^^^^^^^^^

``sum_of_products`` forms plain outer products and applies no swap gate.  Its
entries are therefore the coefficients of :math:`O` in the *interleaved word*
``(out_0, in_0, out_1, in_1, ...)``, the order in which the outer product
stacks the legs, and the MPO tensors inherit that meaning.  The Fock-basis
matrix elements of the same operator are indexed by the *nested word*
``(out_0, ..., out_{L-1}, in_{L-1}, ..., in_0)``.  Reordering one word into
the other costs one swap gate per pair of legs that change relative order,
and the product of those swap gates is exactly the Jordan-Wigner string of
:math:`O`.  The measurement never applies this reordering to the tensors.  It
works in the Fock basis throughout and lets the strings arise as line
crossings in the network, so nothing has to be baked into the MPO.

The Fock-basis MPO can be calculated from the interleaved MPO as follows.  Every bond
leaves its tensor at the upper-left port, passes over the top of that tensor
and enters the next tensor at the lower-left port.  On the way it crosses
exactly one line, the ``in`` leg of the tensor it left; bonds cross neither
each other nor any ``out`` leg.  Bond ``k`` in the picture carries the
network label ``('opb', k)``::

               i0             i1             i2              i3
                |              |              |               |
           +----X----+    +----X----+    +----X----+          |
           |    |    |    |    |    |    |    |    |          |
           |  +-+--+ |    |  +-+--+ |    |  +-+--+ |        +-+--+
           +--| M0 | |    +--| M1 | |    +--| M2 | |        | M3 |
              |    | +--1----|    | +--2----|    | +--3-----|    |
              +-+--+         +-+--+         +-+--+          +-+--+
                |               |               |             |
               o0              o1              o2             o3

           X = swap gate between the bond and the in leg it crosses
           1, 2, 3 = network labels ('opb', 1), ('opb', 2), ('opb', 3)

Each crossing ``X`` is one swap gate, :math:`(-1)^{p(\mathrm{bond})\,p(\mathrm{in})}`
block by block.  Why this reproduces the Jordan-Wigner string: in the block
where site ``j`` carries the local operator charge :math:`q_j`, the bond
leaving tensor ``k`` carries the cumulative charge :math:`\sum_{j \le k} q_j`.
Every term of :math:`O` is charge-neutral, so that is the same parity as
:math:`\sum_{j > k} q_j`, and the crossing at site ``k`` contributes
:math:`(-1)^{p(n_k)\, p(\sum_{j > k} q_j)}` with :math:`n_k` the occupation on
the ``in`` leg of site ``k``.  That is precisely the sign of commuting all
later operators past site ``k``.  Because the charge is resolved sector by
sector on the bond, terms of different fermion parity can share one MPO.

On the lattice the same crossings are routed through the double layer to the
top-left corner of the window, along the string path of plain charged
operators (previous section).  A bond leg has no fixed charge, so its
crossings cannot be folded into the site tensors; they become swap pairs
between the bond leg and the crossed network legs, evaluated block by block
during the contraction.
A bond ``('opb', k)`` connects two sites, so it collects the symmetric
difference of their two string paths: a leg on both paths is swapped twice and
drops out, and the bond ends up swapped only against the legs between its two
sites.  The crossed ket leg is the bare one, before the operator acts, so it
must remain a network leg; this is why MPO tensors require
``separate_layers=True``.


Cut maps and randomized range finding
-------------------------------------

Large windows can be made cheaper by compressing interior cuts with pairs of
CTM half-projectors, passed through ``projectors``.  A cut severs the *fat*
bond between two neighbouring sites, made of one environment bond and the ket
and bra bonds next to it; a half-projector pair :math:`P_A, P_B` squeezes it
into a *thin* bond.  The pairs stored in ``env.proj`` were fitted during CTM
to its corner contractions, not to the window being measured.
:meth:`yastn.tn.fpeps.EnvCTM.measure_nsite_cut_map_oe` provides the
contraction needed to fit them to the window itself.

Cut map
^^^^^^^

With every other cut compressed, the window is a linear map :math:`X` from the
fat bond on one side of the cut to the fat bond on the other side.  The cut
map is

.. math::

    Y = X \, \Omega,

where the probe :math:`\Omega` takes the place of one half-projector and has
a few thin columns.  The partner side is left open, so :math:`Y` has four
legs: the environment, ket and bra legs of the open fat bond, and the probe's
thin leg.

.. code-block:: text

    pair inserted:   A ══ P_A ── thin ── P_B ══ B     ->  a number
    cut map:         A ══  Ω  ── thin      fat ══ B   ->  Y, thin and fat legs open

Closing :math:`Y` with the partner half gives the measurement with the pair
inserted.  With the probe equal to the stored half :math:`P_A`::

    P_A = env.proj[probe_site].hlb                          # stored form (env, ket x bra, thin)
    P_B = env.proj[partner_site].hlt.unfuse_legs(axes=(1,)) # (env, ket, bra, thin)
    Y = env.measure_nsite_cut_map_oe(*ops, sites=sites,
                                     probe_site=probe_site, probe_slot='hlb', probe=P_A,
                                     projectors=other_cuts)
    value = yastn.tensordot(Y, P_B, axes=((0, 1, 2, 3), (0, 1, 2, 3))).to_number()
    # == env.measure_nsite_numerator_exact_oe(*ops, sites=sites, projectors=all_cuts)

The side left open is the probe's partner: the slot with the last letter of
``probe_slot`` flipped (``t`` with ``b``, ``l`` with ``r``) on the neighbouring
site in that letter's direction, e.g. ``'hlt'`` at ``(1, 0)`` for a probe
``'hlb'`` at ``(0, 0)``.  A half and its partner sever the same two bonds, so
the open legs follow from the probe alone.

.. automethod:: yastn.tn.fpeps.EnvCTM.measure_nsite_cut_map_oe

Randomized range finding
^^^^^^^^^^^^^^^^^^^^^^^^

A probe with random columns turns the cut map into a randomized range finder
(N. Halko, P.-G. Martinsson and J. A. Tropp, SIAM Rev. 53, 217 (2011)).  Draw
:math:`\Omega = [\,p \mid G\,]`, with :math:`p` the current half-projector as a
warm start and :math:`G` a random block with a few extra columns per charge
sector.  The columns of :math:`Y = X\Omega` then span the dominant range of
:math:`X`.  An orthonormal basis :math:`U` of that range, truncated with an SVD
of :math:`Y`, gives the new pair :math:`P = U U^\dagger`, inserted at the cut as
the two halves :math:`U^\dagger` and :math:`U`.

The error of any window trace is controlled by the discarded singular values,

.. math::

    |\mathrm{Tr}\, X - \mathrm{Tr}(P X)| \le \sum_{i>r} \sigma_i(X),

whenever the range of :math:`P` contains the top-:math:`r` left singular
subspace of :math:`X`.  The bound needs neither Hermiticity nor positivity of
:math:`X`.  Several maps that share a cut, the norm window and one numerator
window per operator product, can therefore share one pair: stack their
sketches, each scaled by the quantity whose relative error it controls, and
truncate the stacked matrix once.  Every sketch costs one cut-map contraction.

:func:`yastn.tn.fpeps.envs.sketch_proj.sketch_projectors_` sketches every cut of
a projector layout this way, pooling the norm window and all numerator windows
of the patch; the example below spells out the steps for one cut.

.. autofunction:: yastn.tn.fpeps.envs.sketch_proj.sketch_projectors_

Example: sketching one cut
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given an environment ``env`` whose ``env.proj`` holds the CTM projectors, an
MPO ``mpo`` of the window, ``k`` random columns per charge sector, a target
rank ``chi`` and a flag ``two_sided``, the pair of one cut, fitted to the norm
window and one numerator window, is::

    sites = [fpeps.Site(0, 0), fpeps.Site(1, 1)]         # the window
    top, bot = fpeps.Site(0, 0), fpeps.Site(1, 0)        # the cut: 'hlb' at top, its partner 'hlt' at bot
    ctm_pair = {top: ('hlb',), bot: ('hlt',)}

    def probe(half):
        """[half | G]: a CTM half as warm start plus k random columns per charge sector."""
        fat = yastn.leg_product(half.get_legs(0), half.get_legs(1))   # every sector of the fat bond
        thin = half.get_legs(2)
        extra = yastn.Leg(env.config, s=thin.s, t=fat.t, D=[min(k, d) for d in fat.D])
        G = yastn.rand(env.config, legs=[half.get_legs(0), half.get_legs(1), extra], n=half.n)
        return yastn.block({(0,): half, (1,): G}, common_legs=(0, 1))

    # weight of each window: the value it controls
    s_norm = abs(env.measure_nsite_norm_exact_oe(sites=sites, projectors=ctm_pair))
    s_num = abs(env.measure_nsite_numerator_exact_oe(*mpo, sites=sites, projectors=ctm_pair))

    # Y = X . Omega, one cut map per window: probe on the bottom side, top side open
    kw = dict(sites=sites, probe_site=bot, probe_slot='hlt', probe=probe(env.proj[bot].hlt))
    blocks = [env.measure_nsite_cut_map_oe(**kw) / s_norm,
              env.measure_nsite_cut_map_oe(*mpo, **kw) / s_num]

    if two_sided:  # also sketch X^dagger: probe the partner half instead
        kw = dict(sites=sites, probe_site=top, probe_slot='hlb', probe=probe(env.proj[top].hlb))
        blocks += [env.measure_nsite_cut_map_oe(**kw).conj() / s_norm,      # conj(X^T . Omega')
                   env.measure_nsite_cut_map_oe(*mpo, **kw).conj() / s_num]

    # stack the sketches and truncate once
    W = yastn.block({(i,): B for i, B in enumerate(blocks)}, common_legs=(0, 1, 2))
    U, _, _ = yastn.svd_with_truncation(W, axes=((0, 1, 2), 3), sU=env.proj[top].hlb.get_legs(2).s,
                                        nU=True, D_total=chi)

    # the new pair: Q on the open side, Q.conj() on the probe side
    Q = U.conj().fuse_legs(axes=(0, (1, 2), 3))          # stored form (env, ket x bra, thin)
    pair = {top: {'hlb': Q}, bot: {'hlt': Q.conj()}}
    value = (env.measure_nsite_numerator_exact_oe(*mpo, sites=sites, projectors=pair)
             / env.measure_nsite_norm_exact_oe(sites=sites, projectors=pair))

The random block covers every charge sector of the fat bond,
:func:`yastn.leg_product` of the half's two legs, including sectors the CTM
half does not carry; fusing the half's own legs would drop them.  With a probe
that spans the fat bond (``k`` at least the dimension of every sector) and no
truncation, the pair is the identity on the cut and reproduces the
uncompressed values exactly.  Several cuts are sketched one after another,
each with the other cuts closed by their current pairs through
``projectors``.

With ``two_sided = True`` the adjoint map is sketched as well.  The same
contractions with the probe moved to the partner slot return
:math:`Z = X^T\,\Omega'`, since a contraction never conjugates, and
``Z.conj()`` :math:`= X^\dagger\,\bar\Omega'` is a sketch of :math:`X^\dagger`
in the same space as :math:`Y`.  Each adjoint block takes the scale of its
forward partner, because :math:`X` and :math:`X^\dagger` have the same singular
values.  The pair then spans the dominant subspace of :math:`[X, X^\dagger]`,
at twice the number of cut-map contractions.  Part of the rank goes to the
co-range, so at a fixed rank the value is not necessarily more accurate; what
the two-sided sketch buys is the gradient bound of the next subsection.

Which half carries the conjugate matters.  The signatures already force
``U.conj()`` rather than ``U`` onto the open side, since ``U`` inherits its
fat legs from the cut map's open legs.  The data must be conjugated as well:
the kernel inserted at the cut is :math:`\sum_i Q_{ai} \bar Q_{bi}`, whose
acting subspace is the conjugate of the span of :math:`Q`, so :math:`Q` must
carry the conjugated data of :math:`U` for the kept subspace to be the span of
:math:`U`.  ``U.flip_signature()`` has the right signatures but unconjugated
data.  For complex tensors it keeps the wrong subspace and converges much more
slowly with the rank; only at full rank, or for real tensors, do the two
coincide.

Gradients
^^^^^^^^^

The sketched pair enters the measurement as a constant: build it without a
computation graph, under ``torch.no_grad()``, or ``detach()`` its halves.  The
projectors are a device for approximating a quantity that does not depend on
them, and the gradient is taken of the compressed value at fixed :math:`P`,

.. math::

    E_P(A) = \mathrm{Tr}\big(P\,X(A)\big), \qquad
    \frac{\partial E_P}{\partial A}\Big|_P = \mathrm{Tr}\Big(P\,\frac{\partial X}{\partial A}\Big).

The term left out is :math:`\mathrm{Tr}(dP\,X)`.  With :math:`P = U U^\dagger`
and :math:`U^\dagger U = 1`, :math:`dP = (1-P)\,dU\,U^\dagger + U\,dU^\dagger\,(1-P)`,
so

.. math::

    \mathrm{Tr}(dP\,X) = \mathrm{Tr}\big(U^\dagger X\,(1-P)\,dU\big)
                       + \mathrm{Tr}\big(dU^\dagger\,(1-P)\,X\,U\big).

It vanishes when the kept subspace is invariant under both :math:`X` and
:math:`X^\dagger`: exactly at full rank, where :math:`1-P = 0`, and for a
Hermitian map truncated to its dominant eigenvectors, where :math:`P`
maximizes :math:`\mathrm{Tr}(PX)` over projectors of that rank (Ky Fan) and
its first-order variation is zero.  In general it is bounded by two leaks,

.. math::

    |\mathrm{Tr}(dP\,X)| \le \|dU\|\,\big(\|(1-P)\,X\|_1 + \|(1-P)\,X^\dagger\|_1\big),

with :math:`\|\cdot\|_1` the trace norm.  A one-sided sketch captures the range
of :math:`X` and controls only the first leak.  The two-sided sketch captures
the range of :math:`X^\dagger` as well.  With :math:`P` the dominant rank-:math:`r`
subspace of :math:`[X, X^\dagger]`, both leaks are at most the discarded weight
of that stacked matrix, the same weight that bounds the value error,

.. math::

    |\mathrm{Tr}(dP\,X)| \le 2\,\|dU\| \sum_{i>r} \sigma_i\big([X, X^\dagger]\big).

The factor :math:`\|dU\|`, the sensitivity of the kept subspace to the state,
grows when :math:`\sigma_r` and :math:`\sigma_{r+1}` are close.  A randomized
range finder reaches the bound up to its usual oversampling error.

Differentiating through the construction would not make the gradient more
accurate.  The probe is random, the truncation picks a discrete rank, and the
SVD backward carries factors :math:`1/(\sigma_i^2 - \sigma_j^2)` that blow up
for nearly degenerate singular values at the truncation edge.  It would also
keep the graphs of every cut-map contraction of the sketch in memory until the
backward pass.  Fix the random seed of the probe, so that the compressed value
is a reproducible function of the state: a line search must not see a
stochastic objective.
