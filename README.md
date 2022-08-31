# WaveTrain

by *Jerome Riedel, Patrick Gelß, and Burkhard Schmidt*

Freie Universität Berlin, Germany

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

## Installation

After downloading and installing the Python tensor train toolbox [scikit\_tt](https://github.com/PGelss/scikit_tt),
you can download the latest version of **WaveTrain** to your local computer by using the 'git clone' command.  
Note that a [*setup.py*](setup.py) file and a [*setup.cfg*](setup.cfg) file are included in the package. 
To install the package simply enter:

```
git clone https://github.com/PGelss/wave_train.git 
cd wave_train
python setup.py install --user
```

which subsequently allows to download or upload recent changes later by 'git pull' or 'git push', respectively.

Alternatively, you may install the latest version directly from GitHub:

```
pip install git+https://github.com/PGelss/wave_train
```
in which case there is no direct git access.
Later changes can be downloaded by the command 

```
pip install --upgrade git+https://github.com/PGelss/wave_train
```

## Applications

see our work on coupled excitons and phonons [^1]

[^1]: P. Gelß, R. Klein, S. Matera, B. Schmidt, "Solving the Time-Independent Schrödinger Equation for 
Chains of Coupled Excitons and Phonons using Tensor Trains", [J. Chem. Phys. 156 (2), 024109 (2022)](https://doi.org/10.1063/5.0074948) 
