============
Purification
============

The thermal state for a Hamiltonian :math:`H`` and inverse temperature  :math:`\beta = 1/(k_B T)`  is given by  :math:`\rho_{\beta} = \frac{\exp(-\beta H)}{\text{Tr}(\exp(-\beta H))}`,
where Tr denotes the trace operation. Since in tensor networks, pure states are more amenable to proper representation and manipulation, we often choose to embed our thermal density matrix 
in a pure state by adding an ancillary Hilbert space to the physical Hilbert space. So we effectively work with a larger Hilbert space, and then trace out the degrees of freedom corresponding 
to the ancilla to obtain the thermal density matrix. The technique is outlined as follows.

We start with the system at infinite temperature (:math:`\beta=0`) where all states are equally probable. This is described as a maximally mixed density matrix :math:`\rho` in the basis 
:math:`\ket{\psi_{n}}` where the number of states in the physical Hilbert space labelled by :math:`n` runs from :math:`1` to :math:`d`:

:math:`\rho = \frac{\mathds{1}}{d}= \sum_{n} \frac{1}{d} \ket{\psi_{n}}\bra{\psi_{n}}`.

Then we write a purified wave-function :math:`\ket{\psi_{\infty}}` at infinite temperature as a maximally entangled state between the physical and ancillary degrees of freedom where the latter
is introduced using the basis :math:`\ket{\psi_{n'}}`:
:math:`\ket{\psi_{\infty}} = \frac{1}{\sqrt{d}} \sum_{n=n'=1}^{d}\ket{\psi_{n}} \ket{\psi_{n'}}`.

We define the state at finite temperature :math:`\beta` by evolving with :math:`U = \exp(-\frac{\beta}{2}H)`:

:math:`\ket{\psi_{\beta}} = \exp\left(-\frac{\beta}{2} H \right) \ket{\psi_{\infty}} = \frac{1}{\sqrt{d}} \sum_{n=n'=1}^{d}\exp\left(-\frac{\beta}{2} H \right)  \ket{\psi_{n}} \ket{\psi_{n'}}`

Now we take the trace over the ancillary degrees of freedom from the total density matrix :math:`\rho_{tot} = \ket{\psi_{\beta}} \bra{\psi_{\beta}}` to recover the thermal density matrix 
of the physical system:
:math:`\text{Tr}_{n'}\rho_{tot} = \exp(-\beta H) = \rho_{\beta}`.
