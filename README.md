# wave_train

**wave_train** is an open-source software for numerical simulations of chain-like 
quantum systems with nearest-neighbor (NN) interactions only.
(with or without periodic boundary conditions).
This Python package is centered around tensor train (TT, or matrix product) representations of 
quantum-mechanical Hamiltonian operators and (stationary or time-evolving) state vectors.
**wave_train** builds on the Python tensor train toolbox [scikit\_tt](https://github.com/PGelss/), 
which provides efficient construction methods, storage schemes, 
as well as solvers for eigenvalue problems and linear differential equations in the TT format.

**wave_train** comprises solvers for time-independent and time-dependent Schrödinger equations 
employing efficient decompositions to construct low-rank representations. 
Often, the tensor-train ranks of state vectors are found to depend only marginally on the chain length *N*, 
which results in the computational effort growing only slightly more than linearly in *N*, 
thus mitigating the curse of dimensionality.
As a complement to the Python classes for full quantum mechanics, **wave_train** also contains classes for 
fully classical and mixed quantum-classical (Ehrenfest or mean field) dynamics of bipartite systems.
The graphical capabilities allow visualization of quantum dynamics ‘on the fly’, with a choice of 
several different graphical representations based on reduced density matrices.
