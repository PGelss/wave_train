import sys
import numpy as np
import scikit_tt.utils as utl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.evp as evp
from wave_train.dynamics.quant_mech import QuantumMechanics


class TISE(QuantumMechanics):
    """
    Solving the time-independent Schroedinger equation
    for a chain/ring system with NN interactions only.
    Numeric solution based on tensor train representations
    """

    def __init__(self, hamilton, n_levels,
                 solver='als', eigen='eig',
                 ranks=20, repeats=20, conv_eps=1e-8,
                 e_est=0, e_min=0, e_max=1,
                 save_file=None, load_file=None, compare=None):
        """
        Parameters:
        -----------
        hamilton: instance of physical object (quantum Hamiltonian)
            Either one Exciton, Phonon, Coupled or other classes
        n_levels: int
            number of solutions (energy levels) of the TISE
        solver: string, optional
            algorithm to solve eigenproblems of the full system, can be 'als' or 'qe'
        eigen: string, optional
            algorithm to solve eigenproblems of the micro systems, can be 'eig', 'eigs' or 'eigh'
        ranks: int
            ranks of TT representation of state vectors
        repeats: int
            number of iterations in ALS scheme for decomposition
        conv_eps : float, optional
            threshold for detecting convergence of the eigenvalue, default is 1e-8.
            When set to -1, the convergence criterion will not be used.
            Instead, the number of iterations will be controlled by "repeats"
        e_est: float, optional
            find eigenvalues closest to this number
        e_min: float, optional
            lower end of energy plot axis (if exact energies not available!)
        e_max: float, optional
            upper end of energy plot axis (if exact energies not available!)
        save_file: string or 'None', optional
            if not None, data will be saved to mat-file or pickle-file
        load_file: string or 'None'
            if not None, reference data will be loaded from pickle-file
        compare: 3-character-string or 'None'
            How to compare with reference data:
            'pos': Positions (of vibrational degrees of freedom)
            'pop': Populations (of quantum states)
            'psi': Complete state vectors
        """
        
        self.n_levels  = n_levels
        self.solver   = solver
        self.eigen   = eigen
        self.ranks    = ranks
        self.repeats  = repeats
        self.conv_eps  = conv_eps
        self.e_est    = e_est
        self.e_min    = e_min
        self.e_max    = e_max
        self.save_file = save_file
        self.load_file  = load_file
        self.compare    = compare

        # Extra information useful for data file output
        self.name = self.__class__.__name__

        # If available, calculate analytic/exact solutions
        if callable(getattr(hamilton, 'get_exact', None)):
            energies = hamilton.get_exact(n_levels)
            if energies is not None:
                self.exct = energies

        # Fake time steps to make TISE results look like TDSE results
        self.step_size = 1
        if hasattr(self,'exct'):
            self.num_steps = len(self.exct)-1
        else:
            self.num_steps = self.n_levels-1

        # Initialize object of parent class
        QuantumMechanics.__init__(self, hamilton)

        # Pre-allocate arrays for eigen|values|vectors
        if self.solver == 'qe':
            self.eigen_values = np.zeros(self.n_levels)
            self.eigen_vectors = np.zeros([self.hamilton.n_dim**self.hamilton.n_site, self.n_levels])
            self.operator = np.zeros([self.hamilton.n_dim**self.hamilton.n_site, self.hamilton.n_dim**self.hamilton.n_site])

    def __str__(self):
        info = """
-----------------------------------------
Solving the TISE using TT representations
-----------------------------------------

Number of eigenvalues wanted     : {}

Choice of solution method        : {}
Choice of eigensolver            : {}

Ranks of TT representation       : {}
Number of iterations in ALS      : {}
Convergence criterion for ALS    : {}

Find eigenvalues closest to      : {}

Saving generated data to file    : {}
Loading reference data from file : {}
How to compare with reference    : {} 
       """.format(self.num_steps+1,
                  self.solver, self.eigen,
                  self.ranks, self.repeats, self.conv_eps,
                  self.e_est,
                  self.save_file, self.load_file, self.compare)

        return info

    # Use this method for solving the TISE *without* visualization
    # Note the alternative method for solving the TISE *with* visualization in class Visual
    def solve(self):

        # Initialize TISE solver
        self.start_solve()
        
        # Loop over stationary states
        for i in range(self.num_steps+1):
            self.update_solve(i)
                
    def start_solve(self):
        
        print(self)

        # If available, load reference solution from data file
        self.load()

        if self.solver == 'als': # Alternating Linear Scheme to solve eigenproblems for TTs

            # Providing an initial guess for quantum state psi
            # self.psi_guess = tt.ones(self.hamilton.represent.row_dims, [1] * self.hamilton.represent.order, ranks=self.ranks).ortho()
            self.psi_guess = tt.canonical(self.hamilton.represent.row_dims, self.ranks).ortho()

        elif self.solver == 'qe': # Quasi-Exact diagonalization, using full matrix representation instead of TT

            # Get matrix representation of Hamiltonian operator
            self.operator = self.hamilton.represent.matricize()
            print("---------------------------------------------------------")
            print("Calculating quasi-exact eigenvalues from full Hamiltonian")
            print("---------------------------------------------------------")
            print(" ")
            print("Diagonalize a matrix of size : ", self.hamilton.n_dim ** self.hamilton.n_site)
            print(" ")

            # Diagonalize the Hamiltonian matrix
            if self.eigen == 'eig':
                eigen_values, eigen_vectors = np.linalg.eig(self.operator)
            elif self.eigen == 'eigh':
                eigen_values, eigen_vectors = np.linalg.eigh(self.operator)
            else:
                sys.exit('Wrong choice of eigen')

            # Only eigenvalues exceeding e_est, sort in ascending order
            idx1 = [i for i, x in enumerate(eigen_values) if x >= self.e_est]
            idx2 = eigen_values[idx1].argsort()
            idx = [idx1[i] for i in idx2]

            self.eigen_values = eigen_values[idx[0:self.n_levels]]
            self.eigen_vectors = eigen_vectors[:, idx[0:self.n_levels]]  # containing (normalized!) column vectors

            print("List of eigenvalues")
            print(self.eigen_values)
            print (" ")

        # Initialize list of computed eigentensors
        self.previous = []
        
    def update_solve(self,i):

        if self.solver == 'als':  # Alternating linear scheme to solve eigenproblems for TTs

            with utl.timer() as cputime:

                # find eigenvalue closest to e_est
                [energy, self.psi, iterations] = evp.als(self.hamilton.represent, self.psi_guess,
                                                         previous = self.previous, shift=100,
                                                         repeats=self.repeats, conv_eps=self.conv_eps,
                                                         solver=self.eigen, sigma=self.e_est)

                # Shift spectrum of Hamiltonian => getting ready for the next round
                # Use Wielandt deflation technique within ALS iterations
                self.previous.append(self.psi.copy())
                self.e_est = energy

        elif self.solver == 'qe':  # Quasi-exact diagonalization, using full matrix representation of TT

            with utl.timer() as cputime:

                energy = self.eigen_values[i]
                self.psi = self.eigen_vectors[:, i]
                iterations = 0  # fake

        else:
            sys.exit('Wrong choice of solver')

        # Save ground state, needed to fake autocorrelation
        if i == 0:
            self.psi_0 = self.psi

        # Update and print observables
        self.cpu_time = cputime.elapsed
        self.title = self.name + ' (' + self.solver + '): state = ' + str(i) + ', energy = ' + str("%.6f" % energy) + ', CPU = ' + str("%.2f" % self.cpu_time) + ' sec'
        self.observe(i, iterations=iterations)

        # Upon last time step: Print date/time
        # Export object 'self' into a data file
        # Perhaps deleting TT cores to save disk space
        if i == self.num_steps:
            # delattr(self.hamilton.represent, 'cores') # really necessary?
            self.save()
