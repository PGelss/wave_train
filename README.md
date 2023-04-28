# WaveTrain

by *Jerome Riedel, Patrick Gelß, and Burkhard Schmidt*

Freie Universität Berlin, Germany

![WaveTrain-Logo](https://raw.githubusercontent.com/pgelss/wave_train/master/design_logo/wavetrain.jpg)

## Short description

**WaveTrain** is an open-source software for numerical simulations of chain-like 
quantum systems with nearest-neighbor (NN) interactions only
(with or without periodic boundary conditions).
This Python package is centered around tensor train (TT, or matrix product) representations of 
quantum-mechanical Hamiltonian operators and (stationary or time-evolving) state vectors.
**WaveTrain** builds on the Python tensor train toolbox [scikit\_tt](https://github.com/PGelss/scikit_tt), 
which provides efficient construction methods, storage schemes, 
as well as solvers for eigenvalue problems and linear differential equations in the TT format.

**WaveTrain** comprises solvers for time-independent and time-dependent Schrödinger equations 
employing TT decompositions to construct low-rank representations. 
Often, the TT ranks of state vectors are found to depend only marginally on the chain length *N*, 
which results in the computational effort growing only slightly more than linearly in *N*, 
thus mitigating the *curse of dimensionality*.
Thus, **WaveTrain** complements the existing [WavePacket project at SourceForge](https://sourceforge.net/projects/wavepacket/)
which does not offer these advantages but which can be used for general Hamiltonians,
i.e., without restriction to chain-like systems.

As a complement to the Python classes for full quantum mechanics, **WaveTrain** also contains classes for 
fully classical and mixed quantum-classical (Ehrenfest or mean field) dynamics of bipartite 
("slow-fast" and/or "heavy-light") systems.
Moreover, the graphical capabilities allow visualization of quantum dynamics ‘on the fly’, with a choice of 
several different graphical representations based on reduced density matrices.

## Full description

For a detailed description of the WaveTrain software, see our article that appeared in February 2023 at arXiv[^1]. To appear in J. Chem. Phys. 

## Installation
After downloading and installing the Python tensor train toolbox [scikit\_tt](https://github.com/PGelss/scikit_tt),
installation of the **WaveTrain** software package is straightforward
```
pip install git+https://github.com/PGelss/scikit_tt
pip install wave_train 
```
where pip belongs to a Python installation with minimum version requirement 3.7.0.
For a developer installation you can download the latest version of **WaveTrain** to your local computer by using the 'git clone' command.  
```
git clone https://github.com/PGelss/wave_train.git 
cd wave_train
python setup.py install --user
```

## Applications

see our work on solving the TISE[^2] and TDSE[^3] for coupled excitons and phonons

[^1]: J. Riedel, P. Gelß, R. Klein, and B. Schmidt, "WaveTrain: A Python Package for Numerical Quantum Mechanics of Chain-Like Systems Based on Tensor Trains", [arXiv:2302.03725](https://arxiv.org/abs/2302.03725)

[^2]: P. Gelß, R. Klein, S. Matera, B. Schmidt, "Solving the Time-Independent Schrödinger Equation for 
Chains of Coupled Excitons and Phonons using Tensor Trains", [J. Chem. Phys. 156 (2), 024109 (2022)](https://doi.org/10.1063/5.0074948) 

[^3]: P. Gelß, R. Klein, S. Matera, and B. Schmidt, "Quantum Dynamics of Coupled Excitons and Phonons in Chain-Like Systems: Tensor Train Approaches and Higher-Order Propagators", [arXiv:2302.03725](https://arxiv.org/abs/2302.03725)
