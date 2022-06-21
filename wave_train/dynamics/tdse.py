import sys
import math
import numpy as np
from scipy.special import jv
from scipy.linalg import expm
import scikit_tt.utils as utl
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.ode as ode
from scikit_tt.tensor_train import TT
from wave_train.dynamics.quant_mech import QuantumMechanics


class TDSE(QuantumMechanics):
    """
    Solving the time-dependent Schroedinger equation
    for a chain/ring system with NN interactions only.
    Numeric solution based on tensor train representations
    """

    def __init__(self, hamilton, num_steps, step_size, sub_steps=1,
                 solver='se', normalize=0,
                 max_rank=20, repeats=20, threshold=1e-12,
                 save_file=None, load_file=None, compare=None):
        """
        hamilton: instance of physical object (quantum Hamiltonian)
            Either one Exciton, Phonon, Coupled or other classes
        num_steps: int 
            number of (main) time steps
        step_size: float
            size of (main) time steps
        sub_steps: int
            number of sub_steps for each main step
        solver: 2-character-string (optional)
            method of numeric integration: ie, ee, tr, se, lt, sm, yn, kl
        normalize: int (optional)
            whether|how to normalize the solutions, can be 0|1|2
        max_rank: int
            maximum_rank of solution
        repeats: int 
            Maximum number of ALS iterations (not for explicit solvers)
        threshold: float
            threshold for reduced SVD decompositons
        save_file: string or 'None'
            if not None, generated data will be saved to mat-file or pickle-file
        load_file: string or 'None'
            if not None, reference data will be loaded from pickle-file
        compare: 3-character-string or 'None'
            How to compare with reference data:
            'pos': Positions (of vibrational degrees of freedom)
            'pop': Populations (of quantum states)
            'psi': Complete state vectors
        """
        
        self.num_steps  = num_steps
        self.step_size  = step_size
        self.sub_steps  = sub_steps
        if sub_steps is not None:
            self.sub_size   = step_size / sub_steps
        else:
            self.sub_size = step_size
        self.solver     = solver
        self.normalize  = normalize
        self.max_rank   = max_rank
        self.repeats    = repeats
        self.threshold  = threshold
        self.save_file  = save_file
        self.load_file  = load_file
        self.compare    = compare

        # Extra information useful for data file output
        self.name = self.__class__.__name__

        # Initialize object of parent class
        QuantumMechanics.__init__(self, hamilton)

        # Initialize TT cores
        cores = [None] * self.hamilton.n_site


    def __str__(self):

        info = """
-----------------------------------------
Solving the TDSE using TT representations
-----------------------------------------

Number of main steps             : {}  
Size of main steps               : {} 
Number of sub steps              : {} 
Size of sub steps                : {} 

Method of numeric integration    : {}
Normalizing after integration    : {}

Maximum rank of solution         : {} 
Maximum number of ALS iterations : {}
Threshold for reduced SVD        : {} 

Saving generated data to file    : {}
Loading reference data from file : {}
How to compare with reference    : {} 
        """.format(self.num_steps, self.step_size, self.sub_steps, self.sub_size,
                   self.solver, self.normalize,
                   self.max_rank, self.repeats, self.threshold,
                   self.save_file, self.load_file, self.compare)

        return info

    def coherent(self, displace):
        """

        Construct an initial quantum state as a tensor train
        Coherent aka quasi-classical aka Glauber state
        --------------------------------------------

        Input:
            displace: List or array
                displacements of each of the sites

        Output
            psi: instance of TT class
                quantum state vector
        """
        # Check delta vector and convert to numpy array
        if not isinstance(displace, list):
            sys.exit("Wrong input of displacements delta: should be a list")
        if len(displace) != self.hamilton.n_site:
            sys.exit("Number of displacement parameters must be equal to number of sites")
        displace = np.asarray(displace, dtype=np.float64)

        print("-------------------------------------------------------")
        print("Construct a (tensor) product of coherent quantum states")
        print("-------------------------------------------------------")
        print(" ")
        print("Initial displacements : " + str(displace))
        print(" ")

        # Set zeta to coherent state parameter on every site
        zetas = 0.5 / self.hamilton.pos_conv * displace

        cores = [None] * self.hamilton.n_site

        # Loop over cores, i.e. over sites
        print("Eigenvalues of the annihilation operator and corresponding coefficients: ")
        print(" ")
        for j, zeta in enumerate(zetas):
            # construct state based on zeta
            prefct = np.exp(-0.5 * zeta ** 2)
            coeffs = prefct * np.array([zeta ** n / np.sqrt(math.factorial(n)) for n in range(self.hamilton.n_dim)])
            print("zeta=" + str(zeta) + " : " + str(coeffs))

            # construct tensor cores
            cores[j] = np.zeros((1, self.hamilton.n_dim, 1, 1))
            cores[j][0, :, 0, 0] = coeffs
        print (" ")

        # Construct tensor train
        self.psi = TT(cores)
        print("--------------------------------------")
        print("Setting up coherent state in TT format")
        print("--------------------------------------")
        print(self.psi)
        print (" ")

        # Quantize TT representation: decompose 2^n tensor into n 2-tensors
        # makes computations much faster, but perhaps a little less accurate
        # assuming that n_dim is an integer power of two
        if self.hamilton.qtt:
            self.psi = TT.tt2qtt(self.psi, self.hamilton.n_site * [[2] * self.hamilton.l2_dim], self.hamilton.n_site * [[1] * self.hamilton.l2_dim],
                            threshold=1e-12)
            print("---------------------------------------")
            print("Converting coherent state to QTT format")
            print("---------------------------------------")
            print(self.psi)
            print (" ")

    def fundamental(self, coeffs=None):
        """ Construct an initial quantum state as a tensor train.
            Initial fundamental (v=1) excitations given by a vector of coefficients.
            Results in a weighted sum of products, each with a single excitation

            Parameters
            ----------
            coeffs : vector of reals
                coefficients of initial excitation

            Returns
            -------
            psi : instance of TT class
                TT representation of the initial quantum state
        """

        # Default: excitation initially near/at center of chain
        if coeffs is None:
            coeffs = np.zeros(self.hamilton.n_site)
            self.localized = self.hamilton.n_site // 2
            coeffs[self.localized] = 1
        else:
            # Check coefficient vector and normalize
            if not isinstance(coeffs, list):
                sys.exit("Wrong input of initial coefficients: should be a list")
            if len(coeffs) != self.hamilton.n_site:
                sys.exit("Inconsistent length of vector of initial coefficients")
            coeffs /= np.linalg.norm(coeffs)

            # Find site with unit excitation
            loc = np.argwhere(coeffs == 1)

            # If only a single site is initially excited
            if (len(loc)) == 1:
                self.localized = loc[0, 0]

        # Console output
        print("--------------------------------------------------------------------")
        print("Construct a (tensor) product of fundamentally excited quantum states")
        print("--------------------------------------------------------------------")
        print(" ")
        print("Initial coefficients   : " + str(coeffs))
        if hasattr(self, 'localized'):
            print("Localized only at site : ", self.localized)
        else:
            print("Not localized")
        print (" ")

        # loop over entries of coefficient vector
        first = True
        for i in range(self.hamilton.n_site):

            # Only for non-vanishing coefficient
            if coeffs[i] != 0:

                # Initialize TT cores
                cores = [None] * self.hamilton.n_site

                # Loop over cores
                for j in range(self.hamilton.n_site):
                    cores[j] = np.zeros([1, self.hamilton.n_dim, 1, 1])

                    if i == j:
                        cores[j][0, :, 0, 0] = self.hamilton.excited
                    else:
                        cores[j][0, :, 0, 0] = self.hamilton.ground

                # Construct new tensor train
                if first:
                    self.psi = coeffs[i] * TT(cores)
                    first = False

                # Add to existing tensor train
                else:
                    self.psi += coeffs[i] * TT(cores)

        print("-------------------------------------")
        print("Setting up initial state in TT format")
        print("-------------------------------------")
        print(self.psi)
        print (" ")

        # Quantize TT representation: decompose 2^n tensor into n 2-tensors
        # makes computations much faster, but perhaps a little less accurate
        # assuming that n_dim is an integer power of two
        if self.hamilton.qtt:
            self.psi = TT.tt2qtt(self.psi, self.hamilton.n_site * [[2] * self.hamilton.l2_dim], self.hamilton.n_site * [[1] * self.hamilton.l2_dim],
                            threshold=1e-12)
            print ("--------------------------------------")
            print ("Converting initial state to QTT format")
            print ("--------------------------------------")
            print(self.psi)
            print(" ")

    # Use this method for solving the TDSE *without* visualization
    # Note the alternative method for solving the TDSE *with* visualization in class Visual
    def solve(self):
        
        # Initialize TDSE solver
        self.start_solve()
        
        # Loop over time steps: propagation
        for i in range(self.num_steps + 1):
            self.update_solve(i)
                
    def start_solve(self):

        print(self)

        # Initial energy, to be used for plot axis scaling only
        self.e_init = np.real(self.expect(self.hamilton.represent, self.psi, self.hamilton.n_site, self.hamilton.l2_dim))
        self.e_min = self.e_init*0.9
        self.e_max = self.e_init*1.1

        # Analytic solutions in terms of Bessel functions for excitons on an homogenous chain
        if self.hamilton.name == 'Exciton' and self.hamilton.homogen and not self.hamilton.periodic:

            # If only a single site is initially excited
            if hasattr(self, 'localized'):
                self.bessel = np.zeros((self.num_steps + 1, self.hamilton.n_site))

                # Highest order of bessel function required on right and left end
                distUp = self.hamilton.n_site - self.localized - 1
                distLow = self.localized

                # Bessel functions of the first kind, order is the distance
                orders = np.arange(-1 * distLow, distUp + 1)
                orders = [-1 * x if x < 0 else x for x in orders]
                time = np.arange(0, (self.num_steps + 1) * self.step_size, self.step_size)
                tau = 1 / (2 * np.abs(self.hamilton.beta[0]))
                for i, order in enumerate(orders):
                    self.bessel[:, i] = np.square(jv(order, time / tau))

                print(" ")
                print("----------------------------------------------------")
                print("Calculating analytic solution using Bessel functions")
                print("----------------------------------------------------")
                print(" ")
                print("Initially excited site : " + str(self.localized))
                print(" ")

        # If available, load reference solution from data file
        self.load()

        # Initial quantum state: required for auto-correlation function
        self.psi_0 = self.psi

        # Guess a state vector: for implicit solvers only
        if self.solver in ['ie', 'tr']:
            self.psi_guess = tt.ones(self.hamilton.represent.row_dims, [1]*self.hamilton.represent.order, ranks=self.max_rank).ortho()

        # Accounting for the TDSE pre-factor -1j: for splitting propagator only
        if self.solver in ['lt', 'sm', 'yn', 'kl']:

            # Limitations in use of splitting methods
            if self.hamilton.periodic:
                sys.exit('Splitting methods do not work with periodic systems')
            if self.hamilton.qtt:
                sys.exit('Splitting methods do not work with quantized TTs')

            # Multiply SLIM matrices with -ij pre-factor
            if isinstance(self.hamilton.S, list):  # heterogeneous chain
                for i in range(len(self.hamilton.S)):
                    self.hamilton.S[i] = -1j*self.hamilton.S[i]
            else:  # homogeneous chain
                self.hamilton.S = -1j*self.hamilton.S
            if isinstance(self.hamilton.L, list):  # heterogeneous chain
                if self.hamilton.periodic:
                    for i in range(len(self.hamilton.L)):
                        self.hamilton.L[i] = -1j*self.hamilton.L[i]
                else:
                    for i in range(len(self.hamilton.L)-1):
                        self.hamilton.L[i] = -1j*self.hamilton.L[i]
            else:  # homogeneous chain
                self.hamilton.L = -1j*self.hamilton.L

        # Quasi-exact propagation, using matrix exponential
        elif self.solver == 'qe':
            self.operator = self.hamilton.represent.matricize()
            self.psi = (self.psi.matricize()).astype(complex)
            self.psi_0 = (self.psi_0.matricize()).astype(complex)
            print("----------------------------------")
            print("Calculating quasi-exact propagator")
            print("----------------------------------")
            print(" ")
            print("Exponentiate a matrix of size : ", self.hamilton.n_dim ** self.hamilton.n_site)
            print(" ")
            self.propagator = expm (-1j * self.operator * self.step_size)

        # For all propagators except splitting or quasi-exact
        else:  # Accounting for the TDSE pre-factor -1j
            self.operator = -1j * self.hamilton.represent.copy()

        # Set up a vector containing all temporal sub-steps: Needed only for ee, se, ie, tr
        self.step_sizes = self.sub_size * np.ones(self.sub_steps)

        # Symmetric Euler requires storing a previous time step
        if self.solver == 'se':
            self.previous = None

        if self.solver in ['s2', 's4', 's6', 's8']:
            self.previous = None
            order = int(self.solver[1])
            self.op_hod = 2 * self.sub_size * self.operator.copy()
            op_tmp = self.operator.copy()
            for k in range(2, order // 2 + 1):
                op_tmp = op_tmp.dot(self.operator).dot(self.operator)
                self.op_hod += 2 / np.math.factorial(2 * k - 1) * self.sub_size ** (2 * k - 1) * op_tmp
            self.op_hod = self.op_hod.ortho(threshold=self.threshold)

    def update_solve(self, i):

        with utl.timer() as cputime:
            
            # Propagation for i>0
            if i > 0:
                
                if self.solver == 'ee':  # Explicit Euler
                    psi = ode.explicit_euler(self.operator, initial_value=self.psi, step_sizes=self.step_sizes,
                                             threshold=self.threshold, max_rank=self.max_rank, normalize=self.normalize, progress=False)
                elif self.solver == 'se':  # Symmetric Euler (fka 'second order differencing' in quantum dynamics community)
                    psi = ode.symmetric_euler(self.operator, initial_value=self.psi,
                                              step_sizes=self.step_sizes, previous_value=self.previous,
                                              threshold=self.threshold, max_rank=self.max_rank,
                                              normalize=self.normalize, progress=False)
                    self.previous = psi[-2]
                elif self.solver in ['s2', 's4', 's6', 's8']:  # Higher order differencing
                    psi = ode.hod(self.operator, initial_value=self.psi, step_size=self.sub_size,
                                  number_of_steps=self.sub_steps, order=int(self.solver[1]),
                                  previous_value=self.previous, op_hod=self.op_hod,
                                  threshold=self.threshold, max_rank=self.max_rank,
                                  normalize=self.normalize, progress=False)
                    self.previous = psi[-2]
                elif self.solver == 'ie':  # Implicit Euler
                    psi = ode.implicit_euler(self.operator, initial_value=self.psi, initial_guess=self.psi_guess,
                                             step_sizes=self.step_sizes, repeats=self.repeats,
                                             threshold=self.threshold, max_rank=self.max_rank, normalize=self.normalize, progress=False)
                elif self.solver == 'tr':  # Trapezoidal rule
                    psi = ode.trapezoidal_rule(self.operator, initial_value=self.psi, initial_guess=self.psi_guess,
                                               step_sizes=self.step_sizes, repeats=self.repeats,
                                               threshold=self.threshold, max_rank=self.max_rank, normalize=self.normalize, progress=False)
                elif self.solver == 'lt':  # Lie-Trotter splitting
                    psi = ode.lie_splitting(self.hamilton.S, self.hamilton.L, self.hamilton.I, self.hamilton.M,
                                            initial_value=self.psi, step_size=self.sub_size,
                                            number_of_steps=self.sub_steps, threshold=self.threshold,
                                            max_rank=self.max_rank, normalize=self.normalize)
                elif self.solver == 'sm':  # Strang-Marchuk splitting
                    psi = ode.strang_splitting(self.hamilton.S, self.hamilton.L, self.hamilton.I, self.hamilton.M,
                                               initial_value=self.psi, step_size=self.sub_size,
                                               number_of_steps=self.sub_steps, threshold=self.threshold,
                                               max_rank=self.max_rank, normalize=self.normalize)
                elif self.solver == 'yn':  # Yoshida-Neri splitting
                    psi = ode.yoshida_splitting(self.hamilton.S, self.hamilton.L, self.hamilton.I, self.hamilton.M,
                                                initial_value=self.psi, step_size=self.sub_size,
                                                number_of_steps=self.sub_steps, threshold=self.threshold,
                                                max_rank=self.max_rank, normalize=self.normalize)
                elif self.solver == 'kl':  # Kahan-Li splitting
                    psi = ode.kahan_li_splitting(self.hamilton.S, self.hamilton.L, self.hamilton.I, self.hamilton.M,
                                                initial_value=self.psi, step_size=self.sub_size,
                                                number_of_steps=self.sub_steps, threshold=self.threshold,
                                                max_rank=self.max_rank, normalize=self.normalize)

                elif self.solver == 'qe':  # Quasi-exact propagation
                    self.psi = self.propagator @ self.psi

                else:
                    sys.exit('Wrong choice of solver')

                # Prepare for next time step
                if self.solver != 'qe':
                    self.psi = psi[-1]
                    self.psi_guess = self.psi

        # Update and print observables
        self.cpu_time = cputime.elapsed
        self.title = self.name + ' (' + self.solver + '): step = ' + str(i) + ', time = ' + str(i*self.step_size) + ', CPU = ' + str("%.2f" % self.cpu_time) + ' sec'
        self.observe(i)

        # Upon last time step: Print date/time
        # Export object 'self' into a data file
        # Linear regression of energy and norm versus time
        if i == self.num_steps:
            self.linear_regression()
            self.save()
