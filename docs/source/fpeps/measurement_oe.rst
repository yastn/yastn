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

The ket, the operator and the bra of every site stay separate network tensors
(:func:`_build_ketbra_separate`), with the fermionic crossings between them as swap
pairs of the contraction.


Plain charged operators
-----------------------

A product of plain operators :math:`O_{s_1} O_{s_2} \cdots O_{s_n}`, listed in any
order, is measured as an MPO of bond dimension one if any of the operators is charged
(a single :math:`c` or :math:`c^\dagger`): :func:`yastn.tn.mps.product_mpo` of the
operators, its chain the sites in the order listed.  The bond between two neighbouring
operators of the chain carries the total charge of the operators after it, and the
charge travels along the MPO bonds exactly as the bonds of an MPO passed by the caller
do (next section).  Operators listed on the same site are multiplied, with the sign of
bringing them together.  A product of operators of zero charge needs no bonds; they sit
on their sites as they are.


.. _oe-mpo-operators:

Operators as MPO tensors
------------------------

Building the MPO
^^^^^^^^^^^^^^^^

A sum of operator products on one set of sites,

.. math::

    O = \sum_t c_t \, o^{(t)}_{s_0} \, o^{(t)}_{s_1} \cdots o^{(t)}_{s_{L-1}},

can be measured in a single contraction instead of one contraction per term.
Build it with :func:`yastn.tn.mps.generate_mpo` and pass the MPO in place of the
plain operators::

    import yastn.tn.mps as mps

    H = mps.generate_mpo(mps.product_mpo(I, N=len(sites)),
                         [mps.Hterm(coeff, list(range(len(sites))), ops) for coeff, ops in terms])
    value = env.measure_nsite_exact_oe(H, sites=sites)

The chain of the MPO is ``sites`` as listed: position ``k`` of an
:class:`yastn.tn.mps.Hterm` is the operator acting on ``sites[k]``, and ``I`` is the
local identity, which fills the positions a term does not list.  The sites may be
listed in any order, and neighbouring positions of the chain need not be
neighbouring sites of the lattice.

:func:`yastn.tn.mps.generate_mpo` returns the operator in the Fock basis: the
entries of its tensors are the matrix elements of :math:`O`, the string of the
chain among them.  On the chain it is then an ordinary MPO, bonds running
straight from one tensor to the next and crossing nothing::

               i0             i1             i2             i3
                |              |              |              |
              +-+--+         +-+--+         +-+--+         +-+--+
              | M0 |----1----| M1 |----2----| M2 |----3----| M3 |
              +-+--+         +-+--+         +-+--+         +-+--+
                |              |              |              |
               o0             o1             o2             o3

           matrix element  <o0 o1 o2 o3| O |i0 i1 i2 i3>
           1, 2, 3 = network labels ('opb', 1), ('opb', 2), ('opb', 3)

In the window each bond joins its two MPO tensors directly, and it gets the swap
gates that applying the MPO to the ket along a path, as
:meth:`yastn.tn.fpeps.Peps.apply_gate_` does, would produce -- worked out rather
than performed, so the operator is never absorbed into the ket.  The path runs along
the lattice from each site of the chain to the next, and a site it only passes
contributes the swap gates an identity there would, without any tensor being added.
At every step the function the gates use,
:func:`yastn.tn.fpeps._gates_auxiliary.ordering_swaps`, adapts the tensors to a step
that runs against the lattice or the fermionic order; at every site the bond crosses
the legs it would meet on being fused into the ket
(:func:`yastn.tn.fpeps._gates_auxiliary.apply_gate_onsite`).  Two bonds sharing a
lattice bond cross once if they are fused into it in different orders at its two
ends.  A bond has no fixed charge, so these swap gates are evaluated block by block
during the contraction.  The path from one site to the next is a shortest one along
the lattice.

What the measurement requires
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Either plain operators, one per listed site, or one MPO, passed as a single
  :class:`yastn.tn.mps.MpsMpoOBC`; the two kinds are never mixed in one call.
* Plain operators may list their sites in any order, and a site more than once.
  The product is taken as written, :math:`O_{s_1} O_{s_2} \cdots O_{s_n}` with the
  rightmost operator acting first, whatever the lattice's fermionic order.
* The chain of an MPO is ``sites`` as listed: ``H[k]`` acts on ``sites[k]``.  Any
  order works, neighbouring positions of the chain need not be neighbouring sites,
  and each site appears once.
* The measurement applies no sign of its own to an MPO: ``generate_mpo`` puts the
  sign of every term into the MPO, from the order in which that term lists its
  operators, with the same convention as plain operators.

A term may list its sites in any order, independently of the chain and of the other
terms: each operator takes the position of its site in the chain, and
``generate_mpo`` sorts the term by those positions and multiplies it by the sign of
that commutation::

    H = mps.generate_mpo(mps.product_mpo(I, N=len(sites)),
                         [mps.Hterm(coeff, [sites.index(s) for s in term_sites], term_ops)
                          for coeff, term_sites, term_ops in terms])
    value = env.measure_nsite_exact_oe(H, sites=sites)

Operator bonds and unrolling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bond between MPO tensors ``k-1`` and ``k`` is labelled ``('opb', k)``.
It is unrolled like any other bond, e.g. ``unroll={('opb', 1): 2}``, or one
charge sector at a time with
:func:`yastn.tensor.oe_blocksparse.make_sliced_legs`.  The dimension-one bonds
at the two ends of the chain are dropped and have no label.  A product of plain
charged operators has such bonds too, of dimension one, along its sites in the
order listed.  The norm network contains no operator bonds, so these entries
are dropped when the norm is contracted.

Example
^^^^^^^

Hopping in both directions plus density-density interaction on a diagonal pair of
sites of a fermionic PEPS with CTM environment ``env``.  The chain lists the sites
against the lattice's fermionic order, the sites are not neighbours, and the two
hopping terms list their sites in opposite orders::

    import yastn
    import yastn.tn.mps as mps

    ops = yastn.operators.SpinlessFermions(sym='U1')
    c, cp, n, I = ops.c(), ops.cp(), ops.n(), ops.I()
    sites = [(0, 1), (1, 0)]
    terms = [(-1.0, [(0, 1), (1, 0)], [cp, c]),   # c+_(0,1) c_(1,0)
             (-1.0, [(1, 0), (0, 1)], [cp, c]),   # c+_(1,0) c_(0,1)
             (0.5, [(0, 1), (1, 0)], [n, n])]
    H = mps.generate_mpo(mps.product_mpo(I, N=len(sites)),
                         [mps.Hterm(co, [sites.index(s) for s in ss], oo) for co, ss, oo in terms])

    norm = env.measure_nsite_norm_exact_oe(sites=sites)
    num = env.measure_nsite_numerator_exact_oe(H, sites=sites, unroll={('opb', 1): 1})
    value = num / norm

    # the same as the plain measurements, each term with its sites as it lists them
    ref = sum(co * env.measure_nsite_numerator_exact_oe(*oo, sites=ss) for co, ss, oo in terms) / norm
