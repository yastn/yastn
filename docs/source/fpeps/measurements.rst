Expectation values
==================

Expectation values in PEPS are calulated using CTMRG. So after the imaginary or real time evolution,
one must initiate the CTMRG procedure with the peps tensors as the input,

.. autofunction:: yastn.tn.fpeps.ctm._ctmrg

One can stop the CTM after a fixed number of iterations. A stopping criteria can also be set based on 
the convergence of one or more observables or by comparing the singular values of the projectors. 
Once the CTMRG environment tensors are found, it is straightforward to obtain one-site and two-site 
observables using the following functions.

Average of one-site observables for all sites can be calculated using the functionalities

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.one_site_avg

and average of two-point correlators can be calculated using the function 

.. autofunction:: yastn.tn.fpeps.ctm._ctm_observables.nn_avg
