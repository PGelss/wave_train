import sys
import numpy as np
from scikit_tt.tensor_train import TT
from wave_train.hamilton.chain import Chain
from wave_train.hamilton.phonon import Phonon
from wave_train.hamilton.exciton import Exciton


class Coupled(Chain): 
    """
    Coupled electron-phonon dynamics 
    --------------------------------
    
    for a chain of exciton sites, connected by harmonic oscillators
        optionally with periodic boundaries
        optionally with position restraints
        with four different types of exciton-phonon coupling (EPC) models
    Parent class is chain.
    
    About the storage scheme (for N sites)
    --------------------------------------
    
    For quantities connected to single sites (alpha, mass, nu, chi)
        we always have vectors of length N, with indices 0...N-1
    
    For quantities connected to one-sided NN interactions between sites i and i+1 (beta, omg, rho, tau)
        for cyclic systems: vectors of length N, with indices 0...N-1
        for linear systems: vectors of length N-1, with indices 0...N-2
        
    For quantities connected to symmetric NN interactions between sites i and i-1,i+1 (sig)
        for cyclic systems: vectors of length N, with indices 0...N-1
        for linear systems: vectors of length N-2, with indices 0...N-3
        with the following storage scheme:
        sigma_2     <-> sig[0]
        sigma_3     <-> sig[1]
        ...             ...
        sigma_{N-1} <-> sig[N-3]
        sigma_N     <-> sig[N-2] ------ cyclic only
        sigma_1     <-> sig[N-1] ------ cyclic only
    """
    
    def __init__(self, n_site, periodic, homogen,
                 alpha=1, beta=1, eta=0,
                 mass=1, nu=1, omg=1,
                 chi=0, rho=0, sig=0, tau=0):
        """
        Parameters:
        -----------
        n_site: int
            Length of the chain
        periodic: boolean
            Periodic boundary conditions
        homogen: boolean
            Homogeneous chain/ring
        alpha: float
            excitonic site energy
        beta: float
            coupling strength between nearest neighbours
        eta: float
            constant energy offset
        mass: float
            Mass of site in chain
        nu: float
            harmonic frequency nu of position restraints
        omg_: float
            harmonic frequency omega of nearest neighbor interactions
        chi : float
            linear electron-phonon tuning constant, localized
        rho : float
            linear electron-phonon tuning constant, non-symmetric
        sig : float
            linear electron-phonon tuning constant, symmetrized
        tau : float
            linear electron-phonon coupling constant, pair distance
        """
        
        # Construct object of parent class
        Chain.__init__(self, n_site, periodic, homogen)

        # Construct objects of phonon and exciton class
        self.ex = Exciton(n_site, periodic, homogen, alpha, beta, eta)
        self.ph = Phonon(n_site, periodic, homogen, mass, nu, omg)

        # Parameters of a homogeneous chain/ring are converted into a vector
        if homogen:

            if isinstance(chi, list):
                sys.exit("Wrong input of chi parameter: should be a scalar")
            self.chi = np.repeat(chi, n_site)

            if isinstance(rho, list):
                sys.exit("Wrong input of rho parameters: should be a scalar")
            if periodic:
                self.rho = np.repeat(rho, n_site)
            else:
                self.rho = np.repeat(rho, n_site-1)

            if isinstance(sig, list):
                sys.exit("Wrong input of sig parameters: should be a scalar")
            if periodic:
                self.sig = np.repeat(sig, n_site)
            else:
                self.sig = np.repeat(sig, n_site-2)

            if isinstance(tau, list):
                sys.exit("Wrong input of tau parameters: should be a scalar")
            if periodic:
                self.tau = np.repeat(tau, n_site)
            else:
                self.tau = np.repeat(tau, n_site-1)

        # Parameters of an inhomogeneous chain/ring should be given as vectors
        else:

            if len(chi) != n_site:
                sys.exit("Inconsistent length of vector of chi parameters")
            self.chi = chi

            if periodic:
                if len(rho) != n_site:
                    sys.exit("Inconsistent length of vector of rho parameters")
            else:
                if len(rho) != n_site - 1:
                    sys.exit("Inconsistent length of vector of rho parameters")
            self.rho = rho

            if periodic:
                if len(sig) != n_site:
                    sys.exit("Inconsistent length of vector of sig parameters")
            else:
                if len(sig) != n_site - 2:
                    sys.exit("Inconsistent length of vector of sig parameters")
            self.sig = sig

            if periodic:
                if len(tau) != n_site:
                    sys.exit("Inconsistent length of vector of tau parameters")
            else:
                if len(tau) != n_site - 1:
                    sys.exit("Inconsistent length of vector of tau parameters")
            self.tau = tau

        # Print output from superclass and from this class
        print(super().__str__())
        print(self)
        
        # Useful for savemat|pickle output: Name and classification of this class
        self.name = self.__class__.__name__
        self.bipartite = True                # Single class of (quasi-) particles
        self.classical = self.ph.classical   # Classical approximation (2nd class of particles) supported

    def __str__(self):
        if self.homogen:
            chi = self.chi[0]
            rho = self.rho[0]
            if not self.periodic and self.n_site==2:
                sig = 0
            else:
                sig = self.sig[0]
            tau = self.tau[0]
        else:
            chi = self.chi
            rho = self.rho
            if not self.periodic and self.n_site==2:
                sig = 0
            else:
                sig = self.sig
            tau = self.tau

        coupled_str = """
-----------------------------        
Coupling EXCITONS and PHONONS
-----------------------------
 
Linear tuning   constant (chi), localized     = {}
Linear tuning   constant (rho), non-symmetric = {}
Linear tuning   constant (sig), symmetrized   = {}
Linear coupling constant (tau), pair distance = {}
        """.format(chi, rho, sig, tau)

        return coupled_str

    def get_2Q(self, n_dims):
        """"
        Second quantization:
        Formulate Hamiltonian in terms of creation/annihilation operators

        Parameters:
        -----------
        n_basis[0] : int
            dimension of (truncated!) Hilbert space for excitons
        n_basis[1] : int
            dimension of (truncated!) Hilbert space for phonons
        """

        # Size of excitonic and phononic basis sets
        n_ex = n_dims[0]
        n_ph = n_dims[1]
        n_dim = n_ex * n_ph

        # Second quantization for excitons, phonons
        self.ex.get_2Q (n_ex)
        self.ph.get_2Q (n_ph)

        # Adapt conversion factors from phonons
        self.pos_conv = self.ph.pos_conv
        self.mom_conv = self.ph.mom_conv

        # Excitonic Hamiltonian in terms of creation/annihilation operators
        self.ex_raise = np.kron(self.ex.raising,  self.ph.identity)  # raising  operator: excitons
        self.ex_lower = np.kron(self.ex.lowering, self.ph.identity)  # lowering operator: excitons
        self.ex_numbr = np.kron(self.ex.qu_numbr, self.ph.identity)  # number   operator: excitons

        # Phononic Hamiltonian in terms of creation/annihilation operators
        self.ph_raise = np.kron(self.ex.identity, self.ph.raising)   # raising  operator: phonons
        self.ph_lower = np.kron(self.ex.identity, self.ph.lowering)  # lowering operator: phonons
        self.ph_numbr = np.kron(self.ex.identity, self.ph.qu_numbr)  # number   operator: phonons
        self.position = self.ph_raise + self.ph_lower                # position operator: phonons
        self.momentum = self.ph_raise - self.ph_lower                # momentum operator: phonons
        self.pos_squa = self.position @ self.position
        self.mom_squa = self.momentum @ self.momentum

        # Identity operator
        self.identity = np.eye(n_dim)

        # Set up ground and excited state
        # Assuming only excitonic excitation, no phononic excitation
        self.ground     = np.zeros(n_dim)
        self.ground[0]  = 1
        self.excited    = np.zeros(n_dim)
        self.excited[self.n_ph] = 1

    def get_SLIM(self, n_dims):
        """

        Set up SLIM matrices, for use in tensor train representation

        Parameters:
        -----------
        n_basis[0] : int
            dimension of (truncated!) Hilbert space for excitons
        n_basis[1] : int
            dimension of (truncated!) Hilbert space for phonons


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
        n_ex = n_dims[0]
        n_ph = n_dims[1]
        n_dim = n_ex * n_ph

        if n_ex != 2:
            sys.exit("""
--------------------------------
Value of n_ex must be set to two
--------------------------------  
            """)
        self.n_ex = n_ex
        self.n_ph = n_ph
        self.n_dim = n_dim

        # Using second quantization
        self.get_2Q (n_dims)

        # Get parameters of the chain
        n_site = self.n_site

        # Get parameters of excitonic Hamiltonian
        alpha = self.ex.alpha
        beta  = self.ex.beta
        eta   = self.ex.eta

        # Get parameters of phononic Hamiltonian
        nu_E = self.ph.nu_E
        omg_E = self.ph.omg_E

        # Shallow copies of parameters of coupling Hamiltonian
        chi_1 = self.chi.copy()
        rho_1 = self.rho.copy()
        rho_2 = self.rho.copy()
        sig_1 = self.sig.copy()
        sig_2 = self.sig.copy()
        tau_1 = self.tau.copy()
        tau_2 = self.tau.copy()

        # Converting (tilde symbol!) from second quantization (cT+c) to coordinate R
        chi_1 *= self.pos_conv                     # all sites
        if self.periodic:
            rho_1 *= self.pos_conv
            rho_2 *= self.pos_conv
            sig_1 *= self.pos_conv
            sig_2 *= self.pos_conv
            tau_1 *= self.pos_conv
            tau_2 *= self.pos_conv
        else:
            for i in range(n_site-1):              # all sites but last
                rho_1[i] *= self.pos_conv[i+1]
                rho_2[i] *= self.pos_conv[i]
                tau_1[i] *= self.pos_conv[i+1]
                tau_2[i] *= self.pos_conv[i]
            for i in range(n_site-2):              # all sites but first and last
                sig_1[i] *= self.pos_conv[i+2]
                sig_2[i] *= self.pos_conv[i]

        # Set up S,L,I,M operators
        S_1 = self.ex_numbr                        # on site: excitons
        S_2 = self.ph_numbr + self.identity / 2    # on site: phonons
        S_3 = self.ex_numbr @ self.position        # on site: exciton-phonon-tuning (matrix product!)

        L_1 = self.ex_raise                        # NN interaction: exciton transfer
        L_2 = self.ex_lower                        # NN interaction: exciton transfer
        L_3 = self.position                        # NN interaction: phonon coupling
        L_4 = self.ex_numbr                        # NN interaction: ex-ph-tuning
        L_5 = self.position                        # NN interaction: ex-ph-tuning
        if tau_1[0] != 0:                          # NN interaction: ex-ph-coupling
            L_6 = self.ex_raise
            L_7 = self.ex_raise @ self.position
            L_8 = self.ex_lower
            L_9 = self.ex_lower @ self.position

        I = self.identity                          # identity operator

        M_1 = self.ex_lower                        # NN interaction: exciton transfer
        M_2 = self.ex_raise                        # NN interaction: exciton transfer
        M_3 = self.position                        # NN interaction: phonon coupling
        M_4 = self.position                        # NN interaction: ex-ph-tuning
        M_5 = self.ex_numbr                        # NN interaction: ex-ph-tuning
        if tau_1[0] != 0:                          # NN interaction: ex-ph-coupling
            M_6 = self.ex_lower @ self.position
            M_7 = self.ex_lower
            M_8 = self.ex_raise @ self.position
            M_9 = self.ex_raise

        # Due to the use of the efficient frequencies nu_E, omg_E,
        # non-periodic phonon chains appear here as non-homogeneous
        # even though the original masses, nu, omega are all equal
        if not self.periodic:
            self.really_homogen = False

        if self.really_homogen:

            # Set up S operators for SLIM representation
            self.S = alpha[0] * S_1 + nu_E[1] * S_2 + (chi_1[0] - rho_2[0]) * S_3 + eta * I / n_site

            # Set up L operators for SLIM representation
            self.L = np.zeros([n_dim, n_dim, 5 + (tau_1[0] != 0) * 4])
            self.L[:, :, 0] = beta[0] * L_1
            self.L[:, :, 1] = beta[0] * L_2
            self.L[:, :, 2] = - omg_E[0] * L_3
            self.L[:, :, 3] = (rho_1[0] + sig_1[0]) * L_4
            self.L[:, :, 4] = - sig_2[0] * L_5

            if tau_1[0] != 0:
                self.L[:, :, 5] =   tau_1[0] * L_6
                self.L[:, :, 6] = - tau_2[0] * L_7
                self.L[:, :, 7] =   tau_1[0] * L_8
                self.L[:, :, 8] = - tau_2[0] * L_9

            # Set up I operators for SLIM representation
            self.I = I

            # Set up M operators for SLIM representation
            self.M = np.zeros([5 + (tau_1[0] != 0) * 4, n_dim, n_dim])
            self.M[0, :, :] = M_1
            self.M[1, :, :] = M_2
            self.M[2, :, :] = M_3
            self.M[3, :, :] = M_4
            self.M[4, :, :] = M_5

            if tau_1[0] != 0:
                self.M[5, :, :] = M_6
                self.M[6, :, :] = M_7
                self.M[7, :, :] = M_8
                self.M[8, :, :] = M_9

        else:

            # define lists of ndarrays
            self.S = [None] * n_site
            self.L = [None] * n_site
            self.I = [None] * n_site
            self.M = [None] * n_site

            # insert SLIM components
            for i in range(n_site):      # all sites (S)
                self.S[i] = alpha[i] * S_1 + nu_E[i] * S_2 + chi_1[i] * S_3 + eta * I / n_site

            for i in range(n_site - 1):  # all sites but last (S,L)
                self.S[i] -= rho_2[i] * S_3
                self.L[i] = np.zeros([n_dim, n_dim, 5 + (tau_1[i] != 0) * 4])
                self.L[i][:, :, 0] = beta[i] * L_1
                self.L[i][:, :, 1] = beta[i] * L_2
                self.L[i][:, :, 2] = - omg_E[i] * L_3
                self.L[i][:, :, 3] = rho_1[i] * L_4

                if tau_1[i] != 0:
                    self.L[i][:, :, 5] =   tau_1[i] * L_6
                    self.L[i][:, :, 6] = - tau_2[i] * L_7
                    self.L[i][:, :, 7] =   tau_1[i] * L_8
                    self.L[i][:, :, 8] = - tau_2[i] * L_9

            for i in range(1, n_site - 1): # all sites but first and last (L)
                self.L[i][:, :, 3] += sig_1[i-1] * L_4

            for i in range(n_site - 2):  # all sites but last two (L)
                self.L[i][:, :, 4] = -sig_2[i] * L_5

            for i in range(n_site):      # all sites (I)
                self.I[i] = I

            for i in range(1, n_site):   # all sites but first (M)
                self.M[i] = np.zeros([5 + (tau_1[i-1] != 0) * 4, n_dim, n_dim])
                self.M[i][0, :, :] = M_1
                self.M[i][1, :, :] = M_2
                self.M[i][2, :, :] = M_3
                self.M[i][3, :, :] = M_4

                if tau_1[i-1] != 0:
                    self.M[i][5, :, :] = M_6
                    self.M[i][6, :, :] = M_7
                    self.M[i][7, :, :] = M_8
                    self.M[i][8, :, :] = M_9

            for i in range(1, n_site -1 ):  # all sites but first and last (M)
                self.M[i][4, :, :] = M_5

            if self.periodic:

                # on last site
                self.S[n_site - 1] -= rho_2 [n_site - 1] * S_3

                # coupling first site to second site (sig only)
                self.L[0][:,:,3] += sig_1[n_site - 1] * L_4

                # coupling last site to last but one site (sig only)
                self.L[n_site - 2][:,:,4] = -sig_2[n_site-2] * L_5
                self.M[n_site - 1][4,:,:] = M_5

                # coupling last site (L) to first site (M)
                self.L[n_site - 1] = np.zeros([n_dim, n_dim, 5 + (tau_1[n_site-1] != 0) * 4])
                self.L[n_site - 1][:, :, 0] = beta[n_site - 1] * L_1
                self.L[n_site - 1][:, :, 1] = beta[n_site - 1] * L_2
                self.L[n_site - 1][:, :, 2] = - omg_E[n_site - 1] * L_3
                self.L[n_site - 1][:, :, 3] = (rho_1[n_site - 1] + sig_1[n_site - 2]) * L_4
                self.L[n_site - 1][:, :, 4] = - sig_2[n_site - 1] * L_5

                self.M[0] = np.zeros([5 + (tau_1[n_site-1] != 0) * 4, n_dim, n_dim])
                self.M[0][0, :, :] = M_1
                self.M[0][1, :, :] = M_2
                self.M[0][2, :, :] = M_3
                self.M[0][3, :, :] = M_4
                self.M[0][4, :, :] = M_5

                if tau_1[n_site - 1] != 0:
                    self.L[n_site - 1][:, :, 5] =   tau_1[n_site - 1] * L_6
                    self.L[n_site - 1][:, :, 6] = - tau_2[n_site - 1] * L_7
                    self.L[n_site - 1][:, :, 7] =   tau_1[n_site - 1] * L_8
                    self.L[n_site - 1][:, :, 8] = - tau_2[n_site - 1] * L_9
                    self.M[0][5, :, :] = M_6
                    self.M[0][6, :, :] = M_7
                    self.M[0][7, :, :] = M_8
                    self.M[0][8, :, :] = M_9


        print("""
-------------------------
Setting up SLIM operators
-------------------------      

Size of excitonic basis set   = {}
Size of vibrational basis set = {}
Size of combined basis set    = {}
        """.format(self.n_ex, self.n_ph, self.n_dim))
        if self.really_homogen:
            print('shape of S = ', self.S.shape)
            print('shape of L = ', self.L.shape)
            print('shape of I = ', self.I.shape)
            print('shape of M = ', self.M.shape)

    def qu_coupling(self, pos):
        """
        Quantum-classical approximation to exciton-phonon-coupling:
        Effect on the quantum sub-system results in an additional,
        time-dependent excitonic Hamiltonian which is diagonal
        """

        # Restrict to symmetric (sig-type) on-site coupling only
        if any(self.chi != 0):
            sys.exit ('Quantum-classical equations of motion not (yet) for chi-type ex-ph-coupling')
        if any(self.rho != 0):
            sys.exit ('Quantum-classical equations of motion not (yet) for rho-type ex-ph-coupling')
        if any(self.tau != 0):
            sys.exit ('Quantum-classical equations of motion not (yet) for tau-type ex-ph-coupling')

        out = np.zeros(self.n_site)
        out[1:-1] = self.sig[0] * (pos[2:] - pos[0:-2])

        if self.periodic:
            out[0] = self.sig[0] * (pos[1] - pos[-1])
            out[-1] = self.sig[0] * (pos[0] - pos[-2])

        return out

    def cl_coupling(self, psi):
        """
        Quantum-classical approximation to exciton-phonon-coupling:
        Effect on the classical sub-system results in additional
        "forces" acting on the vibrational degrees of freedom
        """

        out = np.zeros(self.n_site)
        out[2:] -= self.sig[0] * np.abs(psi[1:-1]) ** 2
        out[0:-2] += self.sig[0] * np.abs(psi[1:-1]) ** 2
        if self.periodic:
            out[1] -= self.sig[0] * np.abs(psi[0]) ** 2
            out[0] -= self.sig[0] * np.abs(psi[-1]) ** 2
            out[-2] += self.sig[0] * np.abs(psi[-1]) ** 2
            out[-1] += self.sig[0] * np.abs(psi[0]) ** 2

        return out



