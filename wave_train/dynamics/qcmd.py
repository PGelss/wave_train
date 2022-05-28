import sys
import numpy as np
import scikit_tt.utils as utl
from scipy.linalg import expm
from wave_train.dynamics.qu_cl_mech import QuantClassMechanics


class QCMD(QuantClassMechanics):
    """
    Class for quantum-classical dynamics (mean field, Ehrenfest)
    based on solving Davydov's celebrated equations of motion.
    """
    def __init__(self, hamilton, num_steps, step_size, sub_steps,
                 solver='pb', normalize=0,
                 save_file=None, load_file=None, compare=None):
        """
        hamilton: instance of physical object (quantum-classical Hamiltonian for a bipartite system)
            Restricted to classes with attribute bipartite == True
        num_steps: int
            number of (main) time steps
        step_size: float
            size of (main) time steps
        sub_steps: int
            number of sub_steps for each main step
        solver: 2-character-string (optional)
            method of numeric integration: lt, sm, pb
        normalize: int (optional)
            whether|how to normalize the quantum part of the solution, can be 0|1|2
        save_file: string or 'None'
            if not None, generated data will be saved to mat-file or pickle-file
        load_file: string or 'None'
            if not None, reference data will be loaded from pickle-file
        compare: 3-character-string or 'None'
            How top compare with reference data:
            'pos': Positions (of vibrational degrees of freedom)
        """

        # Restricted to bipartite systems
        if not hamilton.bipartite:
            sys.exit('Quantum-classical equations of motion only for bipartite systems')

        if not hamilton.classical:
            sys.exit('Quantum-classical equations of motion only for bipartite systems where one subsystem is classical')

        # Restricted to homogeneous chains
        if not hamilton.homogen:
            sys.exit('Quantum-classical equations of motion not (yet) for inhomogeneous chains|rings')

        self.num_steps = num_steps
        self.step_size = step_size
        self.sub_steps = sub_steps
        self.sub_size  = step_size / sub_steps
        self.solver    = solver
        self.normalize = normalize
        self.save_file = save_file
        self.load_file = load_file
        self.compare   = compare

        # Initialize object of parent class
        QuantClassMechanics.__init__(self, hamilton)

        # Extra information useful for data file output
        self.name = self.__class__.__name__

        # Initialize quantum-classical variables with zeros
        self.psi = np.zeros(self.hamilton.n_site, dtype=complex)
        self.pos = np.zeros(self.hamilton.n_site)
        self.mom = np.zeros(self.hamilton.n_site)

    def __str__(self):
        info = """
---------------------------------------------
Solving quantum-classical equations of motion
---------------------------------------------

Number of main steps             : {}  
Size of main steps               : {} 
Number of sub steps              : {} 
Size of sub steps                : {} 

Method of numeric integration    : {}
Normalizing after integration    : {}

Saving generated data to file    : {}
Loading reference data from file : {}
How to do the comparison         : {} 
            """.format(self.num_steps, self.step_size, self.sub_steps, self.sub_size,
                       self.solver, self.normalize,
                       self.save_file, self.load_file, self.compare)

        return info

    def fundamental(self, coeffs=None, noise=None):
        """ Construct an initial quantum state as a (normalized) state vector
            Initial fundamental (v=1) excitations given by a vector of coefficients.

            Parameters
            ----------
            coeffs : vector of reals
                coefficients of initial excitation

            Returns
            -------
            psi : vector of complex
                vector representation of the initial quantum state
        """

        # Default: excitation initially near/at center of chain
        if coeffs is None:
            coeffs = np.zeros(self.hamilton.n_site)
            coeffs[self.hamilton.n_site // 2] = 1
        else:
            # Check coefficient vector and normalize
            if not isinstance(coeffs, list):
                sys.exit("Wrong input of initial coefficients: should be a list")
            if len(coeffs) != self.hamilton.n_site:
                sys.exit("Inconsistent length of vector of initial coefficients")
            coeffs /= np.linalg.norm(coeffs)
        self.psi = coeffs

        # Console output
        print("-----------------------------------------------")
        print("Construct a fundamentally excited quantum state")
        print("-----------------------------------------------")
        print(" ")
        print("Initial coefficients   : " + str(self.psi))
        print (" ")

        # Adding some random "noise" to the classical patricles' momenta
        if noise is not None:
            self.mom += noise * (np.random.rand(self.hamilton.n_site) - 0.5)
            print("-----------------------------------------------")
            print("Adding some 'noise' to the classical sub-system")
            print("-----------------------------------------------")
            print(" ")
            print("Initial momenta   : " + str(self.mom))
            print (" ")


    # Use this method for solving classical EoM *without* visualization
    def solve(self):

        # Initialize classical EoM solver
        self.start_solve()

        # Loop over time steps: perform integration
        for i in range(self.num_steps+1):
            self.update_solve(i)

    def start_solve(self):

        print (self)

        # If available, load reference solution from data file
        self.load()

        # Initial quantum state: required for auto-correlation function
        self.psi_0 = self.psi

        # Hamiltonian for the quantum subsystem (time-independent)
        vec_a = np.ones (self.hamilton.n_site) * self.hamilton.ex.alpha[0]
        vec_b = np.ones (self.hamilton.n_site-1) * self.hamilton.ex.beta[0]
        self.hamilt_quant = np.diag(vec_a)
        self.hamilt_quant += np.diag(vec_b, -1)
        self.hamilt_quant += np.diag(vec_b, 1)
        if self.hamilton.periodic:
            self.hamilt_quant[0,-1] = self.hamilton.ex.beta[0]
            self.hamilt_quant[-1,0] = self.hamilton.ex.beta[0]

        # Hamiltonian for the quantum-classical coupling (time-dependent)
        self.hamilt_qu_cl = self.hamilton.qu_coupling(self.pos)

        # Initial energy, to be used for plot axis scaling only
        energy_quant = np.real_if_close(np.conjugate(self.psi) @ (self.hamilt_quant @ self.psi))  # H is a matrix
        energy_qu_cl = np.real_if_close(np.conjugate(self.psi) @ (self.hamilt_qu_cl * self.psi))  # H is a vector
        energy_class = self.hamilton.ph.potential(self.pos) + self.hamilton.ph.kinetic(self.mom)

        self.e_init = energy_quant + energy_qu_cl + energy_class
        self.e_min = 0
        self.e_max = 1.2 * self.e_init

        # Propagator from the purely quantum part of the Hamiltonian
        if self.solver == 'lt':  # Lie-Trotter
            self.propag_quant = expm(-1j * self.hamilt_quant * self.sub_size)  # full time step
        elif self.solver in ['sm', 'pb']:  # Strang-Marchuk, Pickaback
            self.propag_quant = expm(-1j * self.hamilt_quant * self.sub_size / 2)  # half time step
        else:
            sys.exit("Allowed solvers are lt (Lie-Trotter), sm (Strang-Marchuk), pb (Pickaback) only")

    def update_solve(self, i):

        with utl.timer() as cputime:

            # Propagation for i>0
            if i > 0:

                if self.solver == 'lt':  # Lie-Trotter
                    self.lie_trotter()
                elif self.solver == 'sm':  # Strang-Marchuk
                    self.strang_marchuk()
                elif self.solver == 'pb':  # Pickaback
                    self.pickaback()

        # Update and print observables
        self.cpu_time = cputime.elapsed
        self.title = self.name + ' (' + self.solver + '): step = ' + str(i) + ', time = ' + str(i * self.step_size) + ', CPU = ' + str("%.7f" % self.cpu_time) + ' sec'
        self.observe(i)

        # Upon last time step: Print date/time
        # Export object 'self' into a data file
        # Linear regression of energy and norm versus time
        if i == self.num_steps:
            self.linear_regression()
            self.save()

    def lie_trotter(self):
        """
        Simple splitting algorithm for quant.-class. mechanics
        (1) Lie-Trotter splitting for quantum part
        (2) Velocity-Verlet for classical part
        """

        # Short-hand notations
        mass = self.hamilton.ph.mass
        force = self.hamilton.ph.force
        dt = self.sub_size

        # Initialize first substep
        psi_old = self.psi
        pos_old = self.pos
        mom_old = self.mom
        frc_old = force(pos_old) + self.hamilton.cl_coupling(psi_old)

        # Loop over substeps
        for k in range(self.sub_steps):

            # Quantum propagator: time-independent, full matrix
            psi_new = self.propag_quant @ psi_old

            # Quantum-classical propagator: time-dependent, diagonal
            self.hamilt_qu_cl = self.hamilton.qu_coupling(pos_old)
            e_class = self.hamilton.ph.potential(pos_old) + self.hamilton.ph.kinetic(mom_old)
            psi_new *= np.exp(-1j * dt * (self.hamilt_qu_cl + e_class) )

            # Classical propagator: Velocity-Verlet for positions|momenta
            pos_new = pos_old + mom_old / mass * dt + 0.5 * frc_old / mass * dt**2
            frc_new = force(pos_new) + self.hamilton.cl_coupling(psi_new)
            mom_new = mom_old + + 0.5 * (frc_old + frc_new) * dt

            # Get ready for next sub-step
            psi_old = psi_new
            pos_old = pos_new
            mom_old = mom_new
            frc_old = frc_new

        # Extract results from last sub-step
        self.psi = psi_new
        self.pos = pos_new
        self.mom = mom_new

    def strang_marchuk(self):
        """
        Simple splitting algorithm for quant.-class. mechanics
        (1) Strang-Marchuk splitting for quantum part
        (2) Strang-Marchuk splitting for classical part (aka leap frog)
        """

        # Short-hand notations
        mass = self.hamilton.ph.mass
        force = self.hamilton.ph.force
        dt = self.sub_size

        # Initialize first substep
        psi_old = self.psi
        pos_old = self.pos
        mom_old = self.mom

        # Loop over substeps
        for k in range(self.sub_steps):

            # Quantum propagator: time-independent, full matrix
            psi_new = self.propag_quant @ psi_old

            # Quantum-classical propagator: time-dependent, diagonal
            self.hamilt_qu_cl = self.hamilton.qu_coupling(pos_old)
            e_class = self.hamilton.ph.potential(pos_old) + self.hamilton.ph.kinetic(mom_old)
            psi_new = np.exp(-1j * dt * (self.hamilt_qu_cl + e_class)) * psi_new

            # Quantum propagator: time-independent, full matrix
            psi_new = self.propag_quant @ psi_new

            # Classical propagator: Strang-Marchuk aka Leap Frog
            pos_new = pos_old + mom_old / mass * dt / 2
            frc_new = force(pos_new) + self.hamilton.cl_coupling(psi_new)
            mom_new = mom_old + frc_new * dt
            pos_new = pos_new + mom_new / mass * dt / 2

            # Get ready for next sub-step
            psi_old = psi_new
            pos_old = pos_new
            mom_old = mom_new

        # Extract results from last sub-step
        self.psi = psi_new
        self.pos = pos_new
        self.mom = mom_new


    def pickaback(self):
        """
        Pickaback algorithm for quant.-class. mechanics
        """
        sys.exit ('Pickaback code still missing')
