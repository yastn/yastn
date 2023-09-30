Expectation values
==================

Expectation values in PEPS require contraction of the lattice. This can be approximately done using CTMRG.

.. autofunction:: yastn.tn.fpeps.ctm._ctmrg

One can stop the CTM after a fixed number of iterations. Stopping criteria can also be set based on
the convergence of one or more observables or by comparing the singular values of the projectors.
Once the CTMRG environment tensors are found, it is straightforward to obtain one-site and two-site
observables using the following functions.

One-site observables for all lattice sites can be calculated using the function

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.one_site_dict

and all nearest neighbor two-point correlators can be calculated using the function

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.nn_exp_dict
