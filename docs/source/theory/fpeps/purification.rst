============
Purification
============

The thermal state for a Hamiltonian :math:`H` and inverse temperature  :math:`\beta = 1/(k_B T)`  is given by  :math:`\rho_{\beta} = \frac{\exp(-\beta H)}{\text{Tr}(\exp(-\beta H))}`,
where `Tr` denotes the trace operation. Since in tensor networks, pure states are more amenable to proper representation and manipulation, we often choose to embed our thermal density matrix
in a pure state by adding an ancillary Hilbert space to the physical Hilbert space. The thermal density matrix is obtained tracing out the ancilla degrees of freedom. The technique is outlined as follows.

We start with the system at infinite temperature (:math:`\beta=0`) where all states are equally probable. This is described as a maximally mixed density matrix :math:`\rho_0`.
With the local basis :math:`\ket{e_{n}}` of dimension  :math:`d` (we assume for simplicity that a full Hilbert space of a many-body system is a product of identical local Hilbert spaces)

:math:`\rho_0 = \prod_{sites} \sum_{n} \frac{1}{d} \ket{e_{n}}\bra{e_{n}}`.

Then we write a purified wave-function :math:`\ket{\psi_{0}}` at infinite temperature as a maximally entangled state between the physical and ancillary degrees of freedom where the latter
is introduced using the basis :math:`\ket{e_{n'}}`:
:math:`\ket{\psi_{0}} = \prod_{sites} \frac{1}{\sqrt{d}} \sum_{n=1}^{d}\ket{e_{n}} \ket{e_{n}}`.

We define the state at finite temperature :math:`\beta` by evolving with :math:`U = \exp(-\frac{\beta}{2}H)` acting on physical degrees of freedom:

:math:`\ket{\psi_{\beta}} = \exp\left(-\frac{\beta}{2} H \right) \ket{\psi_{0}}`

Now we take the trace over the ancillary degrees of freedom from the total density matrix :math:`\rho_{tot} = \ket{\psi_{\beta}} \bra{\psi_{\beta}}` to recover the thermal density matrix
of the physical system:
:math:`\text{Tr}_{ancillas} \rho_{tot} = \exp(-\beta H) = \rho_{\beta}`.

In :code:`yastn.tn.fpeps`, each PEPS tensor describing purification has a leg corresponding to physical space and an ancilla leg.
During numerical simulations, they are fused together, with Hamiltonian acting on physical degrees of freedom augmented with identity operator acting on ancillas.
