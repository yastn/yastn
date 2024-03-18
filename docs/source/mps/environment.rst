
Environments
============

Many of the algorithms described in previous sections require MPS/MPOs' product contractions.
Those are supported by environment classes, initialized by :meth:`Env()<yastn.tn.mps.Env>` function.

.. autofunction:: yastn.tn.mps.Env

.. autofunction:: yastn.tn.mps.MpsMpoOBC.on_bra

Environment classes
-------------------

They inherit from the EnvParent class and contain, among others, the following methods:

.. autoclass:: yastn.tn.mps._env.EnvParent
   :members: setup_, clear_site_, factor, measure, update_env_, Heff0, Heff1, Heff2, project_ket_on_bra_1, project_ket_on_bra_2
