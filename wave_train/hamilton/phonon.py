import sys
import numpy as np
from wave_train.hamilton.chain import Chain


class Phonon(Chain):
    """
    Phonon dynamics in one dimension
    ---------------------------------
    
    for a linear chain (or ring system) of harmonic oscillators
        optionally with periodic boundaries
        optionally with position restraints
        
    Parent class is chain.

    Reference
    ---------
        Wikipedia : Quantum harmonic oscillator
            https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator

    """

    def __init__(self, n_site, periodic, homogen, mass=1, nu=1, omg=1):
        """
        Parameters:
        -----------

        n_site: int 
            Length of chain 
        periodic: Boolean
            Periodic boundary conditions
        homogen: boolean
            Homogeneous chain/ring
        mass: float
            Mass of particles
        nu: float
            harmonic frequency nu of position restraints
        omg: float
            harmonic frequency omega of nearest neighbor interactions
        """

        # Construct object of parent class
        Chain.__init__(self, n_site, periodic, homogen)

        # Parameters of a homogeneous chain/ring are converted from scalar to vector
        if homogen:
            if isinstance(mass, list):
                sys.exit("Wrong input of mass parameter: should be a scalar")
            self.mass = np.repeat(mass, n_site)

            if isinstance(nu, list):
                sys.exit("Wrong input of nu parameter: should be a scalar")
            self.nu = np.repeat(nu, n_site)

            if isinstance(omg, list):
                sys.exit("Wrong input of omega parameter: should be a scalar")
            if periodic:
                self.omg = np.repeat(omg, n_site)
            else:
                self.omg = np.repeat(omg, n_site - 1)

        # Parameters of an inhomogeneous chain/ring should be given as vectors
        else:
            if len(mass) != n_site:
                sys.exit("Inconsistent length of vector of mass parameters")
            self.mass = mass

            if len(nu) != n_site:
                sys.exit("Inconsistent length of vector of nu parameters")
            self.nu = nu

            if periodic:
                if len(omg) != n_site:
                    sys.exit("Inconsistent length of vector of omega parameters")
            else:
                if len(omg) != n_site - 1:
                    sys.exit("Inconsistent length of vector of omega parameters")
            self.omg = omg

        # Reduced masses
        if periodic:
            self.red_mass = np.zeros(n_site)
            for i in range(n_site):
                if i == n_site - 1:
                    self.red_mass[i] = self.mass[i] * self.mass[0] / (self.mass[i] + self.mass[0])
                else:
                    self.red_mass[i] = self.mass[i] * self.mass[i + 1] / (self.mass[i] + self.mass[i + 1])
        else:
            self.red_mass = np.zeros(n_site - 1)
            for i in range(n_site - 1):
                self.red_mass[i] = self.mass[i] * self.mass[i + 1] / (self.mass[i] + self.mass[i + 1])

        # Print output from superclass and from this class
        print(super().__str__())
        print(self)

        # Useful for savemat|pickle output: Name and classification of this class
        self.name = self.__class__.__name__
        self.bipartite = False  # Single class of (quasi-) particles
        self.classical = True   # Classical approximation supported

    def __str__(self):
        if self.homogen:
            mass = self.mass[0]
            nu = self.nu[0]
            omg = self.omg[0]
        else:
            mass = self.mass
            nu = self.nu
            omg = self.omg

        phonon_str = """
-----------------------
Hamiltonian for PHONONS
-----------------------

Particle masses                                = {}
Harmonic frequencies (nu, position restraints) = {}
Harmonic frequencies (omg, nearest neighbours)  = {}
Reduced masses                                  = {}
        """.format(mass, nu, omg, self.red_mass)

        return phonon_str

    def get_2Q(self, n_dim=8):
        """"
        Second quantization:
        Formulate Hamiltonian in terms of creation/annihiliation operators
        """

        # Call corresponding method from superclass Chain
        super().get_2Q(n_dim)

        # Effective frequencies: single sites
        self.nu_E = np.zeros(self.n_site)
        for i in range(self.n_site):
            if i == 0:
                if self.periodic:
                    self.nu_E[i] = np.sqrt(
                        self.nu[i] ** 2 + self.red_mass[-1] / self.mass[i] * self.omg[-1] ** 2 + self.red_mass[i] /
                        self.mass[i] * self.omg[i] ** 2)
                else:
                    self.nu_E[i] = np.sqrt(self.nu[i] ** 2 + + self.red_mass[i] / self.mass[i] * self.omg[i] ** 2)
            elif i == self.n_site - 1:
                if self.periodic:
                    self.nu_E[i] = np.sqrt(
                        self.nu[i] ** 2 + self.red_mass[i - 1] / self.mass[i] * self.omg[i - 1] ** 2 + self.red_mass[
                            i] / self.mass[i] * self.omg[i] ** 2)
                else:
                    self.nu_E[i] = np.sqrt(
                        self.nu[i] ** 2 + self.red_mass[i - 1] / self.mass[i] * self.omg[i - 1] ** 2)
            else:
                self.nu_E[i] = np.sqrt(
                    self.nu[i] ** 2 + self.red_mass[i - 1] / self.mass[i] * self.omg[i - 1] ** 2 + self.red_mass[i] /
                    self.mass[i] * self.omg[i] ** 2)

        # Effective frequencies: NN pairs of sites
        if self.periodic:
            self.omg_E = np.zeros(self.n_site)
            for i in range(self.n_site):
                if i == self.n_site - 1:
                    self.omg_E[i] = self.red_mass[i] * self.omg[i] ** 2 / (
                                2 * np.sqrt(self.mass[i] * self.mass[0] * self.nu_E[i] * self.nu_E[0]))
                else:
                    self.omg_E[i] = self.red_mass[i] * self.omg[i] ** 2 / (
                                2 * np.sqrt(self.mass[i] * self.mass[i + 1] * self.nu_E[i] * self.nu_E[i + 1]))
        else:
            self.omg_E = np.zeros(self.n_site - 1)
            for i in range(self.n_site - 1):
                self.omg_E[i] = self.red_mass[i] * self.omg[i] ** 2 / (
                            2 * np.sqrt(self.mass[i] * self.mass[i + 1] * self.nu_E[i] * self.nu_E[i + 1]))

        # Conversion factors from 2nd quantization to position and momentum
        self.pos_conv = np.sqrt(1 / (2 * self.mass * self.nu_E))
        self.mom_conv = 1j * np.sqrt(self.mass * self.nu_E / 2)

        # Set up ground and first excited state
        self.ground = np.zeros(n_dim)
        self.ground[0] = 1
        self.excited = np.zeros(n_dim)
        self.excited[1] = 1

    def get_SLIM(self, n_dim=8):
        """
        Set up SLIM matrices and operators of second quantization,
        mainly for use in tensor train representation

        Parameters:
        -----------
        n_dim: int
            dimension of truncated(!) vibrational Hilbert space

        Returns
        -------
        S: ndarray or list of ndarrays
            single-site components of SLIM decomposition
        L: ndarray or list of ndarrays
            left two-site components of SLIM decomposition
        I: ndarray or list of ndarrays
            identity components of SLIM decomposition
        M: ndarray or list of ndarrays
            right two-site components of SLIM decomposition

        Reference:
        Gelss/Klus/Matera/Schuette : Nearest-neighbor interaction systems in the tensor-train format
        https://doi.org/10.1016/j.jcp.2017.04.007
        """

        # Size of basis set
        self.n_dim = n_dim

        # Using second quantization
        self.get_2Q(self.n_dim)

        # Set up S,L,I,M operators
        S = self.qu_numbr + self.identity / 2
        L = self.position
        I = self.identity
        M = self.position

        # Due to the use of the efficient frequencies nu_E, omg_E,
        # non-periodic phonon chains appear here as non-homogeneous
        # even though the original masses, nu, omega are all equal
        if not self.periodic:
            self.really_homogen = False

        if self.really_homogen:

            # define ndarrays
            self.S = + self.nu_E[0] * S
            self.L = - self.omg_E[0] * L
            self.I = I
            self.M = M

        else:

            # define lists of ndarrays
            self.S = [None] * self.n_site
            self.L = [None] * self.n_site
            self.I = [None] * self.n_site
            self.M = [None] * self.n_site

            # insert SLIM components
            for i in range(self.n_site):  # all sites (S)
                self.S[i] = + self.nu_E[i] * S

            for i in range(self.n_site - 1):  # all sites but last (L)
                self.L[i] = - self.omg_E[i] * L

            for i in range(self.n_site):  # all sites (I)
                self.I[i] = I

            for i in range(1, self.n_site):  # all sites but first (M)
                self.M[i] = M

            if self.periodic:  # coupling last site (L) to first site (M)
                self.L[self.n_site - 1] = - self.omg_E[self.n_site - 1] * L
                self.M[0] = M

        print("""
-------------------------
Setting up SLIM operators
-------------------------      

Size of vibrational basis set = {}
        """.format(self.n_dim))

        if self.really_homogen:
            print('shape of S = ', self.S.shape)
            print('shape of L = ', self.L.shape)
            print('shape of I = ', self.I.shape)
            print('shape of M = ', self.M.shape)

    def get_exact (self, n_levels):
        """
        Compute (almost) exact phononic energy levels for homogeneous phononic systems,
        i.e. analytic or semi-analytic solutions of the underlying TISE for
        cyclic or linear systems, respectively
        
        Parameters:
        -----------
        n_levels: int
            number of exact energies to be computed

        Returns
        -------
        energies: [float]
            list of exact energy levels (if not too many)

        Reference:
        ----------
        Wikipedia : Quantum harmonic oscillator
        https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator
        """

        # Currently this method works only for homogeneous (ring or chain) systems
        if not self.homogen:
            return

        # Get necessary parameters
        nu = self.nu[0]
        omg = self.omg[0]
        N = self.n_site

        if self.periodic:  # Cyclic systems

            # Index j is an even integer running from 0 to 2N-2
            j = np.linspace(0, 2 * N - 2, N)

            # Phonon wave numbers assuming lattice constant a=1
            qa = np.pi * j / N

            # Formula for harmonic phonon frequencies from Wikipedia
            # Compensate for use of reduced(!) masses in Hamiltonian
            # Here with additional position restraints on each particle
            freq = np.array([np.sqrt(nu ** 2 + omg ** 2 * (1 - np.cos(qa[i]))) for i in range(N)])

        else:  # Linear systems

            # Setup Hessian of potential energy function
            H = self.hess_pot()

            # Normal mode analysis
            eigval, eigvec = np.linalg.eig(H)
            freq = np.sqrt(eigval)

        # Maximum energy, maximum quantum numbers
        e_max = np.max(freq) * 3  # always sufficient?
        n_max = (e_max / freq + 1).astype(int)

        # Total number of energy levels
        n_total = 1
        for n in n_max:
            n_total = n_total * int(n)
        # n_total = np.prod(n_max)       # yields negative numbers if n_max large
        do_all = n_total < 10 ** 7

        print('--------------------------------------------------------')
        if self.periodic:
            print('Calculating phononic energy levels analytically')
        else:
            print('Calculating phononic energy levels semi-analytically')
        print('--------------------------------------------------------')
        print(' ')
        print('Phonon frequencies                      : ', str(freq))
        print('Maximal energy threshold = 3*max(freq)  : ', str(e_max))
        print('Maximal quantum numbers below threshold : ', str(n_max))
        print('Total number of states below threshold  : ', str(n_total))
        print('Getting all(!) energies below threshold : ', str(do_all))
        print(' ')

        if do_all:  # all combinations below threshold energy

            energies = np.zeros(n_total)

            # compute quantum energy levels
            for n in range(n_total):
                # Convert flat index n into a tuple of quantum numbers.
                n_q = np.array(np.unravel_index(n, n_max))

                # Uncoupled harmonic oscillators
                energies[n] += np.dot(n_q + 0.5, freq)

        else:  # Only single and double excitations

            print("Calculate zero point energy")
            zpe = sum(freq) / 2 * np.ones(1)
            energies = zpe * np.ones(1)

            print("Appending single excitations")
            energies = np.append(energies, freq + zpe)

            print("Appending double excitations")
            for i in range(0, N):
                for j in range(i, N):
                    energies = np.append(energies, freq[i] + freq[j] + zpe)

            print("Appending triple excitations")
            for i in range(0, N):
                for j in range(i, N):
                    for k in range(j, N):
                        energies = np.append(energies, freq[i] + freq[j] + freq[k] + zpe)

        # sorting the energies in ascending order
        idx = np.argsort(energies)
        energies = energies[idx[:n_levels]]

        return energies

    def potential(self, q):
        """
        Calculates the potential energy function based on the
        positions of each of the particles in the (cyclic or linear) chain
        """

        # Nearest-neighbor forces
        k_n = self.red_mass * self.omg ** 2
        pot = np.sum([k_n[i] * (q[i + 1] - q[i]) ** 2
                      for i in range(self.n_site - 1)]) / 2
        if self.periodic:  # periodic boundary condition
            pot += k_n[-1] * (q[0] - q[-1]) ** 2 / 2

        # Restraining interactions
        k_r = self.mass * self.nu ** 2
        pot += np.sum(k_r * q ** 2) / 2

        return pot

    def kinetic(self, p):
        """
        Calculates the kinetic energy function based on the momenta
        of each of the particles in the (cyclic or linear) chain
        """
        return np.sum(p ** 2 / self.mass) / 2

    def force(self, q):
        """
        Calculates the forces acting on each of the particles
        based on their positions in the (cyclic or linear) chain,
        i.e., the negative gradient of the potential energy function
        """

        force = np.zeros(self.n_site)

        # Nearest-neighbor forces
        k_n = self.red_mass * self.omg ** 2

        # First site
        force[0] = k_n[0] * (q[1] - q[0])  # from right neighbour
        if self.periodic:  # periodic boundary conditions
            force[0] -= k_n[-1] * (q[0] - q[-1]) # from "left" neighbour

        # Inner sites
        for i in range(1,self.n_site-1):
            force[i] = k_n[i] * (q[i + 1] - q[i])  # from right neighbour
            force[i] -= k_n[i - 1] * (q[i] - q[i - 1])  # from left neighbour

        # Last site
        force[-1] = -1 * k_n[-2] * (q[-1] - q[-2])  # from left neighbour
        if self.periodic:  # periodic boundary conditions
            force[-1] += k_n[-1] * (q[0] - q[-1])  # from "right" neighbour

        # Restraining forces
        k_r = self.mass * self.nu ** 2
        force -= k_r * q

        return force

    def hess_pot(self):
        """
        Hessian (matrix of 2nd derivatives) of the potential energy function
        with restraining and NN oscillators (frequency parameters nu and omg)
        """

        if not self.homogen:
            sys.exit('Hessian not (yet) available for non-homogeneous systems')

        n = self.n_site
        fc1 = self.mass[0] * self.nu[0] ** 2      # force constant of restraining oscillators
        fc2 = self.red_mass[0] * self.omg[0] ** 2  # force constant of NN oscillators (reduced mass!)

        dia = fc1 + 2 * fc2              # diagonal contribution
        off = np.ones(n - 1) * fc2       # off-diagonal contribution
        h = np.eye(n) * dia              # diagonal
        if self.periodic:
            h[0, -1] = -fc2              # upper right
            h[-1, 0] = -fc2              # lower left
        else:
            h[0, 0] -= fc2               # "left" boundary
            h[-1, -1] -= fc2             # "right" boundary
        h -= np.diag(off, +1)            # super - diagonal
        h -= np.diag(off, -1)            # sub - diagonal

        return h

    def hess_kin(self):
        """
        Hessian (matrix of 2nd derivatives) of the kinetic energy function: P**2 / (2*M)
        """

        h = np.eye(self.n_site) / self.mass          # diagonal matrix

        return h
