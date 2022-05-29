# WaveTrain

by *Jerome Riedel, Patrick Gelß, and Burkhard Schmidt*

Freie Universität Berlin, Germany

## Short description

**WaveTrain** is an open-source software for numerical simulations of chain-like 
quantum systems with nearest-neighbor (NN) interactions only.
(with or without periodic boundary conditions).
This Python package is centered around tensor train (TT, or matrix product) representations of 
quantum-mechanical Hamiltonian operators and (stationary or time-evolving) state vectors.
**WaveTrain** builds on the Python tensor train toolbox [scikit\_tt](https://github.com/PGelss/scikit_tt), 
which provides efficient construction methods, storage schemes, 
as well as solvers for eigenvalue problems and linear differential equations in the TT format.

**WaveTrain** comprises solvers for time-independent and time-dependent Schrödinger equations 
employing efficient decompositions to construct low-rank representations. 
Often, the tensor-train ranks of state vectors are found to depend only marginally on the chain length *N*, 
which results in the computational effort growing only slightly more than linearly in *N*, 
thus mitigating the curse of dimensionality.
Hence, **WaveTrain** complements the existing [WavePacket project](https://sourceforge.net/projects/wavepacket/)
which does not offer these advantages but which can be used for general Hamiltonians,
i.e., without restriction to chain-like systems.

As a complement to the Python classes for full quantum mechanics, **WaveTrain** also contains classes for 
fully classical and mixed quantum-classical (Ehrenfest or mean field) dynamics of bipartite systems.
The graphical capabilities allow visualization of quantum dynamics ‘on the fly’, with a choice of 
several different graphical representations based on reduced density matrices.

## Installation

to be written ...

## References

[^1] Solving the Time-Independent Schrödinger Equation for Chains of Coupled Excitons and Phonons using Tensor Trains
P. Gelß, R. Klein, S. Matera, B. Schmidt
[J. Chem. Phys. 156 (2), 024109 (2022)](https://doi.org/10.1063/5.0074948) 
