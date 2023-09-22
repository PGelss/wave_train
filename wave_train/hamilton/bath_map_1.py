import sys
import numpy as np
from wave_train.hamilton.chain import Chain


class Bath_Map_1(Chain):

    """
    Hamiltonian for a two level system coupled to a bath
    using mapping of bath degrees of freedom to a chain
    
    Parent class is chain.
    """
    
    def __init__(self, n_site, eta, s, omega_c, omega_0):
        """
        Parameters:
        -----------
        n_site: int
            Length of the chain
        eta: float
            coupling between TLS and first bath site
        s: float
            type of spectral density function: s<1 sub-ohmic, s=1 ohmic, s>1 super-ohmic
        omega_c: float
            cut-off frequency of spectral density function
        omega_0: float
            eigenfrequency of the TLS
        """
        
        # Initialize object of parent class
        Chain.__init__(self, n_site, False, False)
        
        self.eta = eta
        self.s = s
        self.omega_c = omega_c
        self.omega_0 = omega_0

        # Print output from superclass and from this class
        print(super().__str__())
        print(self)
        
        # Useful for savemat|pickle output: Name and classification of this class
        self.name = self.__class__.__name__
        self.bipartite = False  # Single class of (quasi-) particles
        self.classical = False  # Classical approximation not supported

    def __str__(self):
        eta = self.eta
        s = self.s
        omega_c = self.omega_c
        omega_0 = self.omega_0
        
        boson_str = """
----------------------------------------------------
Hamiltonian for a two level system coupled to a bath
using mapping of bath degrees of freedom to a chain 
----------------------------------------------------  

Coupling between TLS and first bath site (eta)   = {}
Type of spectral density function (s)  = {}
Cut-off frequency of spectral density function (omega_c) = {}
Eigenfrequency of the TLS (omega_0) = {}
        """.format(eta, s, omega_c, omega_0)

        return boson_str

    def get_2Q(self, n_dim=2):
        """"
        Second quantization:
        Formulate Hamiltonian in terms of creation/annihiliation operators
        """

        # Call corresponding method from superclass Chain
        super().get_2Q(n_dim)

        # Set up ground and excited state
        # Assuming that the two-level-system is in a Schroedinger CAT state
        self.ground = np.zeros(n_dim)
        self.ground[0] = 1
        self.excited = np.zeros(n_dim)
        self.excited[0] = np.sqrt(0.5)
        self.excited[1] = np.sqrt(0.5)
        
        
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

        self.n_dim = 3
        
        # Using second quantization
        self.get_2Q (self.n_dim)
        
        sigmax = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        sigmay = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        sigmaz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        #nbtotal = self.qu_numbr# np.diag([0, 1, 2])
        #b = self.lowering#np.diag([1,np.sqrt(2)], k=1)
        #bdagger = self.raising# np.diag([1,np.sqrt(2)], k=-1)
        #Q = self.position#b + bdagger
            
        # bond lists
        tls_chain_bond = [self.eta] + [0]*(self.n_site-2)
        chain_bond = [0] + [self.omega_c*(1+i)*(1+i+self.s)/((2+2*i+self.s)*(3+2*i+self.s))*np.sqrt((3+2*i+self.s)/(1+2*i+self.s)) for i in range(self.n_site-2)]
        tls_onsite = [self.omega_0] + [0]*(self.n_site-1)
        cavity_onsite = [0] + [self.omega_c/2*(1 + self.s**2/((self.s+2*i)*(2+self.s+2*i))) for i in range(self.n_site-1)]


        # define lists of ndarrays
        self.S = [-self.omega_0*sigmaz] + [cavity_onsite[i]*self.qu_numbr for i in range(1,self.n_site)]
        self.L = [0.5*np.stack((self.eta*sigmax, self.eta*self.eta*sigmax), axis=-1)] + [0.5*np.stack((chain_bond[i]*self.lowering,chain_bond[i]*self.raising), axis=-1) for i in range(1,self.n_site-1)] + [None]
        self.I = [np.eye(3) for i in range(self.n_site)]
        self.M = [None, np.stack((self.position, self.position))] + [np.stack((self.raising,self.lowering)) for i in range(2,self.n_site)]



# OBSERVABLES
# 
# given a quantum state psi as TT, do the following:
#
# sigmax = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
# sigmay = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
# sigmaz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
#
# exp_vals = np.zeros([3, psi.order])
# for i in range(psi.order):
#     psi_tmp = psi.copy()
#     psi_tmp.cores[i] = np.einsum('ij,kjlm->kilm', sigmax, psi_tmp.cores[i])
#     exp_vals[0, i] = psi.transpose(conjugate=True)@psi_tmp
#     psi_tmp = psi.copy()
#     psi_tmp.cores[i] = np.einsum('ij,kjlm->kilm', sigmay, psi_tmp.cores[i])
#     exp_vals[1, i] = psi.transpose(conjugate=True)@psi_tmp
#     psi_tmp = psi.copy()
#     psi_tmp.cores[i] = np.einsum('ij,kjlm->kilm', sigmaz, psi_tmp.cores[i])
#     exp_vals[2, i] = psi.transpose(conjugate=True)@psi_tmp
    

