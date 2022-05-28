import sys
import numpy as np
from scipy.linalg import expm
import scikit_tt.utils as utl
from wave_train.dynamics.class_mech import ClassicalMechanics

class CEoM(ClassicalMechanics):
    """
    Class for calculating classical trajectories of sites in a chain/ring
    based on solving Newton's or Hamilton's classical equations of motion.
    """
    def __init__(self, hamilton, num_steps, step_size, sub_steps=1,
                 solver='vv',
                 save_file=None, load_file=None, compare=None):
        """
        hamilton: instance of physical object (classical Hamiltonian)
            Restricted to classes with attribute classical == True
        num_steps: int
            number of (main) time steps
        step_size: float
            size of (main) time steps
        sub_steps: int
            number of sub_steps for each main step
        solver: string (optional)
            method of numeric integration
        save_file: string or 'None'
            if not None, generated data will be saved to mat-file or pickle-file
        load_file: string or 'None'
            if not None, reference data will be loaded from pickle-file
        compare: string (3 characters) or 'None'
            How to compare with reference data:
            # TODO: specify choice
        """

        # Restrict to classes with attribute classical == True
        if not hamilton.classical:
            sys.exit("Classical equations of motion not supported for Hamiltonian: " + hamilton.name)

        self.num_steps = num_steps
        self.step_size = step_size
        self.sub_steps = sub_steps
        if sub_steps is not None:
            self.sub_size = step_size / sub_steps
        else:
            self.sub_size = step_size
        self.solver    = solver
        self.save_file = save_file
        self.load_file = load_file
        self.compare   = compare

        # Initialize object of parent class
        ClassicalMechanics.__init__(self, hamilton)

        # Extra information useful for data file output
        self.name = self.__class__.__name__

        # Initialize positions and momenta with zeros
        self.pos = np.zeros(self.hamilton.n_site)
        self.mom = np.zeros(self.hamilton.n_site)

    def __str__(self):
            info = """
-------------------------------------
Solving classical equations of motion
-------------------------------------

Number of main steps             : {}  
Size of main steps               : {} 
Number of sub steps              : {} 
Size of sub steps                : {} 

Method of numeric integration    : {}

Saving generated data to file    : {}
Loading reference data from file : {}
How to compare with reference    : {} 
            """.format(self.num_steps, self.step_size, self.sub_steps, self.sub_size,
                       self.solver,
                       self.save_file, self.load_file, self.compare)

            return info

    def coherent(self, displace):
        """
        Displaces each of the particles in the chain|ring
        from their equilibrium positions by given lengths
        Note the analogy with coherent states in quantum dynamics
        """

        if not isinstance(displace, list):
            sys.exit("Wrong input of displacements : should be a list")
        if len(displace) != self.hamilton.n_site:
            sys.exit("Number of displacements must be equal to number of sites")

        # Set displacements of particles and set corresponding momenta to zero
        self.pos = np.asarray(displace, dtype=np.float64)
        self.mom = np.zeros(self.hamilton.n_site)

        # Console output
        print("------------------------------------")
        print("Set up an initial classical state")
        print("------------------------------------")
        print(" ")
        print("Initial displacements : " + str(self.pos))
        print(" ")
        print("Initial velocities    : " + str(self.mom))
        print(" ")

    # Use this method for solving classical EoM *without* visualization
    def solve(self):

        # Initialize classical EoM solver
        self.start_solve()

        # Loop over time steps: perform integration
        for i in range(self.num_steps+1):
            self.update_solve(i)

    def start_solve(self):

        print (self)

        # Initial energy, to be used for plot axis scaling only
        self.e_init = self.hamilton.potential(self.pos) + \
                      self.hamilton.kinetic(self.mom)
        self.e_min = 0
        self.e_max = 1.2 * self.e_init

        # Quasi-exact propagation, using matrix exponential of the (combined) Hessians
        if self.solver == 'qe':
            f = self.hamilton.hess_pot()
            g = self.hamilton.hess_kin()
            z = np.zeros([self.hamilton.n_site, self.hamilton.n_site])
            m = np.block([[z, g],[-f, z]])

            print("----------------------------------")
            print("Calculating quasi-exact propagator")
            print("----------------------------------")
            print(" ")
            print("Exponentiate a matrix of size : ", 2 * self.hamilton.n_site)
            print(" ")
            self.exp_mat = expm(m*self.step_size)

    def update_solve(self, i):

        with utl.timer() as cputime:

            # Propagation for i>0
            if i>0:

                if self.solver == 'lf':  # Leap-Frog
                    self.leap_frog()
                elif self.solver == 'rk':  # Runge-Kutta
                    self.runge_kutta()
                elif self.solver == 'vv':  # Velocity-Verlet
                    self.velocity_verlet()
                elif self.solver == 'qe':  # quasi-exact
                    self.quasi_exact()
                else:
                    sys.exit("Allowed solvers are lf (Leap-Frog), rk (Runge-Kutta), vv (Velocity-Verlet), qe (quasi-exact) only")

        # Update and print observables
        self.cpu_time = cputime.elapsed
        self.title = self.name + ' (' + self.solver + '): step = ' + str(i) + ', time = ' + str(i * self.step_size) + ', CPU = ' + str("%.6f" % self.cpu_time) + ' sec'
        self.observe(i)

        # Upon last time step: Print date/time
        # Export object 'self' into a data file
        # Linear regression of energy and norm versus time
        if i == self.num_steps:
            self.linear_regression()
            self.save()

    def leap_frog(self):
        """
        Leap frog algorithm
        """
        mass = self.hamilton.mass
        force = self.hamilton.force
        dt = self.sub_size

        pos_old = self.pos
        mom_old = self.mom

        for k in range(self.sub_steps):
            pos_mid = pos_old + 0.5 * dt * mom_old / mass
            mom_new = mom_old + 1.0 * dt * force(pos_mid)
            pos_new = pos_mid + 0.5 * dt * mom_new / mass

            pos_old = pos_new
            mom_old = mom_new

        self.pos = pos_new
        self.mom = mom_new

    def velocity_verlet(self):
        """
        Velocity-Verlet algorithm
        """
        mass = self.hamilton.mass
        force = self.hamilton.force
        dt = self.sub_size

        pos_old = self.pos
        mom_old = self.mom
        frc_old = force(pos_old)

        for k in range(self.sub_steps):
            pos_new = pos_old + dt * mom_old / mass + 0.5 * dt**2 * frc_old / mass
            frc_new = force(pos_new)
            mom_new = mom_old + 0.5 * dt * (frc_old + frc_new)

            pos_old = pos_new
            mom_old = mom_new
            frc_old = frc_new

        self.pos = pos_new
        self.mom = mom_new

    def runge_kutta(self):
        """
        Fourth order Runge-Kutta integration scheme
        https://en.wikipedia.org/wiki/Runge-Kutta_methods
        """
        mass = self.hamilton.mass
        force = self.hamilton.force
        dt = self.sub_size

        x_old = self.pos
        v_old = self.mom / mass

        for k in range(self.sub_steps):
            k1 = dt * force(x_old) / mass
            l1 = dt * v_old
            k2 = dt * force(x_old + 0.5 * l1) / mass
            l2 = dt * (v_old + 0.5 * k1)
            k3 = dt * force(x_old + 0.5 * l2) / mass
            l3 = dt * (v_old + 0.5 * k2)
            k4 = dt * force(x_old + l3) / mass
            l4 = dt * (v_old + k3)

            x_new = x_old + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
            v_new = v_old + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            x_old = x_new
            v_old = v_new

        self.pos = x_new
        self.mom = v_new * mass

    def quasi_exact(self):
        """
        Assuming potential and kinetic energy to be quadratic,
        time-independent functions, Hamilton's equation are solved
        in phase space using matrix exponentials
        Pro: Quasi-analytic, quasi-exact
        Con: Limit on number of sites, due to effort of matrix exponentiation
        """

        pos_old = self.pos
        mom_old = self.mom

        gam_old = np.concatenate((pos_old,mom_old), axis=None)
        gam_new = self.exp_mat @ gam_old

        self.pos = gam_new[:self.hamilton.n_site]
        self.mom = gam_new[self.hamilton.n_site:]

