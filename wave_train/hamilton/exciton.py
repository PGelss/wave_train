import sys
import numpy as np
from wave_train.hamilton.chain import Chain


class Exciton(Chain):

    """
    Exciton dynamics in one dimension
    ---------------------------------
    
    Linear chain of electronic sites with periodic boundaries
    Model consists of site energies (alpha) and nearest neighbor coupling (beta)
    
    Parent class is chain.
    """
    
    def __init__(self, n_site, periodic, homogen, alpha=1, beta=1, eta=0):
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
        """
        
        # Initialize object of parent class
        Chain.__init__(self, n_site, periodic, homogen)

        # Parameters of a homogeneous chain/ring are converted from scalar to vector
        if homogen:
            if isinstance(alpha,list):
                sys.exit("Wrong input of alpha parameter: should be a scalar")
            self.alpha = np.repeat(alpha, n_site)

            if isinstance(beta,list):
                sys.exit("Wrong input of beta parameters: should be a scalar")
            if periodic:
                self.beta = np.repeat(beta, n_site)
            else:
                self.beta = np.repeat(beta, n_site-1)

        # Parameters of an inhomogeneous chain/ring should be given as vectors
        else:
            if len(alpha) != n_site:
                sys.exit("Inconsistent length of vector of alpha parameters")
            self.alpha = alpha

            if periodic:
                if len(beta) != n_site:
                    sys.exit("Inconsistent length of vector of beta parameters")
            else:
                if len(beta) != n_site - 1:
                    sys.exit("Inconsistent length of vector of beta parameters")
            self.beta = beta

        # Constant energy offset
        self.eta = eta

        # Print output from superclass and from this class
        print(super().__str__())
        print(self)
        
        # Useful for savemat|pickle output: Name and classification of this class
        self.name = self.__class__.__name__
        self.bipartite = False  # Single class of (quasi-) particles
        self.classical = False  # Classical approximation not supported

    def __str__(self):
        if self.homogen:
            alpha = self.alpha[0]
            beta = self.beta[0]
        else:
            alpha = self.alpha
            beta = self.beta
        eta = self.eta
        exciton_str = """
------------------------
Hamiltonian for EXCITONS
------------------------      

Excitonic site energy (alpha) = {}
NN coupling strength (beta)   = {}
Constant energy offset (eta)  = {}
        """.format(alpha, beta, eta)

        return exciton_str

    def get_2Q(self, n_dim=2):
        """"
        Second quantization:
        Formulate Hamiltonian in terms of creation/annihiliation operators
        """

        # Call corresponding method from superclass Chain
        super().get_2Q(n_dim)

        # Set up ground and first excited state
        self.ground = np.zeros(n_dim)
        self.ground[0] = 1
        self.excited = np.zeros(n_dim)
        self.excited[1] = 1

    def get_SLIM(self, n_dim=2):
        """
        Set up SLIM matrices and operators of second quantization,
        mainly for use in tensor train representation

        Parameters:
        -----------
        n_dim: int
            dimension of truncated(!) Hilbert space

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
        if n_dim != 2:
            sys.exit("""
---------------------------------
Value of n_dim must be set to two
---------------------------------  
                            """)
        self.n_dim = n_dim

        # Using second quantization
        self.get_2Q (self.n_dim)

        # Set up S,L,I,M operators
        S   = self.qu_numbr
        L_1 = self.raising
        L_2 = self.lowering
        I   = self.identity
        M_1 = self.lowering
        M_2 = self.raising

        if self.homogen:

            # define ndarrays
            self.S = self.alpha[0] * S + self.eta * I / self.n_site
            self.L = self.beta[0] * np.stack((L_1, L_2), axis=-1)
            self.I = I
            self.M = np.stack((M_1, M_2))

        else:

            # define lists of ndarrays
            self.S = [None] * self.n_site
            self.L = [None] * self.n_site
            self.I = [None] * self.n_site
            self.M = [None] * self.n_site

            # insert SLIM components
            for i in range(self.n_site):      # all sites (S)
                self.S[i] = self.alpha[i] * S + self.eta * I / self.n_site

            for i in range(self.n_site - 1):  # all sites but last (L)
                self.L[i] = self.beta[i] * np.stack((L_1, L_2), axis=-1)
                # self.L[i][:, :, 0] = self.beta[i] * L_1
                # self.L[i][:, :, 1] = self.beta[i] * L_2

            for i in range(self.n_site):      # all sites (I)
                self.I[i] = I

            for i in range(1, self.n_site):   # all sites but first (M)
                self.M[i] = np.stack((M_1, M_2))
                # self.M[i][0, :, :] = M_1
                # self.M[i][1, :, :] = M_2

            if self.periodic:            # coupling last site (L) to first site (M)
                self.L[self.n_site - 1] = self.beta[self.n_site - 1] * np.stack((L_1, L_2), axis=-1)
                self.M[0] = np.stack((M_1, M_2))

        print("""
-------------------------
Setting up SLIM operators
-------------------------      

Size of excitonic basis set = {}
        """.format(self.n_dim))
        if self.homogen:
            print('shape of S = ', self.S.shape)
            print('shape of L = ', self.L.shape)
            print('shape of I = ', self.I.shape)
            print('shape of M = ', self.M.shape)

    def get_exact (self, n_levels):
        """Compute analytic (exact) phononic energy levels for homogeneous systems,
        i.e. analytic solutions of the underlying TISE.
        So far, only up to two quanta of excitation
        
        Parameters:
        -----------
        n_levels: int
            number of exact energies to be computed

        Return:
        ------
        energies: list of floats
            list of exact energy levels

        Reference:
        ----------
        Wikipedia article : Hueckel method
            https://en.wikipedia.org/wiki/H%C3%BCckel_method
        Z Hu, G. S. Engels, S. Kais : J. Chem. Phys. 148, 204307 (2018)
            https://doi.org/10.1063/1.5026116
        """

        # This method works only for homogeneous chains|rings
        if not self.homogen:
            return

        # Get necessary parameters
        n_site = self.n_site
        alpha = self.alpha[0]
        beta = self.beta[0]
        eta = self.eta

        # index j is an even integer running from 0 to 2N-2
        j = np.linspace(0, 2*n_site-2, n_site) 
        
        # ground state: zero quanta of excitation
        energies = eta * np.ones(1)
        n_excite = 0

        if self.periodic: # Cyclic systems (rings)
        
            # states with one quantum of excitation
            if n_levels>1:
                n_excite = 1
                ak = np.pi * j / n_site # wavenumber for lattice constant a=1
                ec = alpha + 2 * beta * np.cos(ak)  # component energy
                energies = np.append(energies, ec + eta)

            # states with two quanta of excitation
            if n_levels>1+n_site:
                n_excite = 2
                ak = np.pi * (j+1) / n_site # wavenumber for lattice constant a=1
                ec = alpha + 2 * beta * np.cos(ak) # component energy
                ee,ff = np.meshgrid(ec,ec) # coordinate matrices from coordinate vectors
                em = ee + ff # summing up two component energies
                for i in range(1,n_site): # loop over all superdiagonals
                    energies = np.append(energies, np.diag(em,i)+ eta )
                    
            # states with more than two quanta not (yet?) implemented
            if n_levels>1+sum(range(n_site+1)):
                print ('Wrong choice of num_levels : ' + str(n_levels))
                sys.exit(1)
                
        else: # Linear systems (chains)
                
            # states with one quantum of excitation
            if n_levels>1:
                n_excite = 1
                ak = np.pi * (j/2+1) / (n_site+1) 
                ec = alpha + 2 * beta * np.cos(ak) 
                energies = np.append(energies, ec)
                
            # states with more than one quantum not (yet?) implemented
            if n_levels>1+n_site:
                print ('Wrong choice of num_levels : ' + str(n_levels))
                sys.exit(1)
        
        # sorting the energies in ascending order
        energies = np.sort(energies)

        # truncate unnecessray energy levels
        energies = energies[:n_levels]

        print("""
------------------------------------------------
Calculating excitonic energy levels analytically 
------------------------------------------------

Number of energy levels desired  : {}
Maximum number of excitations    : {}
                        """.format(len(energies), n_excite))

        return energies

