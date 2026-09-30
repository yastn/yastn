CTM options
===========

Every option accepted by the CTM routines is a field of one of three dataclasses. They are the
single place each default is defined, and they are what the loose keyword arguments below are
normalized into.

.. list-table::
    :widths: 22 78
    :header-rows: 1

    * - Object
      - Covers
    * - :class:`yastn.tn.fpeps.envs.CTMOpts`
      - one CTM sweep: moves, projector method, truncation, convergence, execution
    * - :class:`yastn.tn.fpeps.envs.SIOpts`
      - the subspace-iteration projector mode, nested under ``CTMOpts.opts_si``
    * - :class:`yastn.tn.fpeps.envs.FixedPointOpts`
      - the forward / fixed-point pair plus the Neumann backward controls


Two ways to pass them
---------------------

Loose keyword arguments remain the common case::

    env.ctmrg_(opts_svd={'D_total': chi}, max_sweeps=200, corner_tol=1e-8)

Alternatively, build the options once and reuse them. This is what the ``opts=`` keyword is for::

    from yastn.tn.fpeps.envs import make_ctm_opts

    opts = make_ctm_opts(opts_svd={'D_total': chi}, max_sweeps=200, corner_tol=1e-8)
    env_a.ctmrg_(opts=opts)
    env_b.ctmrg_(opts=opts, max_sweeps=10)      # same, but ten sweeps

Explicit keywords win over ``opts``, which in turn wins over ``env.default_opts`` if that is set.
Anything left unspecified falls back to the dataclass default. Deriving never mutates the original:
options are frozen, and nested bundles are merged into the base rather than replacing it, so
``make_ctm_opts(opts, opts_svd={'policy': 'fullrank'})`` keeps the rest of ``opts.opts_svd``.


Unknown options are rejected
----------------------------

A name that is not an option raises ``YastnError`` listing the accepted fields::

    >>> env.ctmrg_(opts_svd={'D_total': 8}, max_sweep=10)
    YastnError: CTM option 'max_sweep' not recognized. Accepted: checkpoint_move, conv_check, ...

Some older spellings are still accepted and normalized: ``opts_svd_ctm`` for ``opts_svd``,
``iterator`` for ``iterator_step``, ``asvr_iterations`` for ``adaptive_spectrum_iterations``, and
the refinement acronyms ``'cwo'``, ``'asvr'``, ``'rds'`` for their spelled-out names.


From a configuration file, with command-line overrides
------------------------------------------------------

The options are plain stdlib dataclasses holding config-representable types, so they round-trip
through nested dictionaries. No configuration framework is required or assumed::

    import yaml
    from yastn.tn.fpeps.envs import CTMOpts, from_dict, override, to_dict

    opts = from_dict(CTMOpts, yaml.safe_load(open('ctm.yaml')))     # defaults <- file
    opts = override(opts, {'opts_svd.D_total': 64,                  #          <- CLI
                           'max_sweeps': 200,
                           'opts_si.niter': 3})
    env.ctmrg_(opts=opts)

:func:`~yastn.tn.fpeps.envs.override` addresses fields by dotted path, including keys inside
``opts_svd`` and fields of the nested ``SIOpts``. Each layer is validated the same way, so an
unknown name in the file or on the command line is an error. Unlike the
factories -- where ``None`` means "not supplied", so a function signature can forward its defaults
harmlessly -- an override of ``None`` is an explicit assignment and does clear the field.

:func:`~yastn.tn.fpeps.envs.argspec` yields ``(dotted_name, type, default, help, choices)`` for
every field, so a command-line interface can be generated from the schema rather than restating it::

    for name, typ, default, help_, choices in argspec(CTMOpts):
        parser.add_argument(f'--{name}', default=default, help=help_,
                            **({'choices': choices} if choices else {}))

One limit is worth knowing: a callable has no representation in a configuration file. ``conv_check``
and a ``mask_f`` inside ``opts_svd`` round-trip in memory but not through YAML or JSON, so they must
be set from code. :func:`~yastn.tn.fpeps.envs.to_dict` emits them as objects.


Reference
---------

.. autoclass:: yastn.tn.fpeps.envs.CTMOpts
    :members: svd_kwargs, si_trunc_kwargs, si_enabled

.. autoclass:: yastn.tn.fpeps.envs.SIOpts

.. autoclass:: yastn.tn.fpeps.envs.FixedPointOpts
    :members: neumann_max_iter, neumann_tol, from_legacy_dicts

.. autofunction:: yastn.tn.fpeps.envs.make_ctm_opts

.. autofunction:: yastn.tn.fpeps.envs.make_si_opts

.. autofunction:: yastn.tn.fpeps.envs.make_fixed_point_opts

.. autofunction:: yastn.tn.fpeps.envs.to_dict

.. autofunction:: yastn.tn.fpeps.envs.from_dict

.. autofunction:: yastn.tn.fpeps.envs.override

.. autofunction:: yastn.tn.fpeps.envs.argspec

.. seealso::

    :doc:`environment_ctm` for the CTMRG iteration and its SI mode, and
    :doc:`fixed_point` for the differentiable fixed point.
