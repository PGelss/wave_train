import sys
import numpy as np
from datetime import datetime
from scikit_tt.tensor_train import TT

class Chain:
    """
    Parent class of classes Exciton, Phonon, Coupled, and upcoming classes
    Contains number of sites, whether the system is homogeneous or not
    and whether periodic boundary conditions (chain or ring) are used.
    Contains also a method to set up tensor train representations
    of quantum-mechanical Hamiltonians from given SLIM matrices.
    """
    def __init__(self, n_site, periodic, homogen):
        """
        Parameters:
            n_site: int
                Length of the chain
            periodic: boolean
                Periodic boundary conditions
            homogen: boolean
                Homogeneous chain/ring
            really_homogen: boolean
                Really a homogeneous chain/ring

        """

        if n_site < 2:
            print ("At least two sites required")
            sys.exit(0)

        self.n_site = n_site
        self.periodic = periodic
        self.homogen = homogen

        self.really_homogen = self.homogen

    def __str__(self):

        # Current date and time formatted as: dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        chain_str = """
-------------------------------
One dimensional chain (or ring)
-------------------------------      

Current date and time          : {}
Number of sites in chain/ring  : {}
Periodic boundary conditions   : {}
Homogeneous chain/ring         : {}
        """.format(dt_string, self.n_site, self.periodic, self.homogen)
        return chain_str

    def get_2Q(self, n_dim):
        """"
        Second quantization:
        Formulate Hamiltonian in terms of creation/annihilation operators
        """

        # Formulate Hamiltonian in terms of creation/annihilation operators
        vec_sqrt = np.sqrt([i for i in range(1, n_dim)])
        self.raising  = np.diag(vec_sqrt, -1)         # raising (creation) operator
        self.lowering = np.diag(vec_sqrt, 1)          # lowering (annihilation) operator
        self.identity = np.eye(n_dim)                 # identity operator
        self.qu_numbr = self.raising @ self.lowering  # number operator
        self.position = self.raising + self.lowering  # position operator
        self.momentum = self.raising - self.lowering  # momentum operator
        self.pos_squa = self.position @ self.position
        self.mom_squa = self.momentum @ self.momentum

    def get_TT(self, n_basis, qtt=False):
        """
        Set up tensor train representation of quantum-mechanical Hamiltonian
        from given SLIM matrices

        Parameters:
        -----------

        n_basis: integer (or vector thereof)
            size of basis set(s)
        qtt: logical
            whether TT representations are quantized, assuming that n_dim is an integer power of two

        Returns
        -------
        hamilt: instance of TT class
            TT representation of Hamiltonian

        Reference: 
        Gelss/Klus/Matera/Schuette : Nearest-neighbor interaction systems in the tensor-train format
        https://doi.org/10.1016/j.jcp.2017.04.007
        """

        # optionally use quantized TT representation
        self.qtt = qtt

        # compute components for TT representation
        self.get_SLIM(n_basis)

        n_dim = self.n_dim
        n_site = self.n_site

        # Construct supercores of tensor train (TT) representation 
        cores = [None] * n_site

        if self.really_homogen:

            # Reshape L and M two 3D arrays if necessary (i.e., if only one two-site interaction)
            if len(self.L.shape) == 2:
                self.L = self.L[:,:,None]
                self.M = self.M[None, :, :]

            # Get number(s) of nearest neighbor pair (L-M) interactions and core size(s)
            n_pair, n_size = self.core_size()

            # First supercore
            cores[0] = np.zeros([1, n_dim, n_dim, n_size])        #  row vector
            cores[0][0, :, :, 0] = self.S
            cores[0][0, :, :, 1:1+n_pair] = self.L
            cores[0][0, :, :, 1+n_pair] = self.I
            if self.periodic:
                cores[0][0, :, :, 2+n_pair:2+2*n_pair] = self.M.transpose([1, 2, 0])
            
            # Last supercore
            cores[-1] = np.zeros([n_size, n_dim, n_dim, 1])       # column vector
            cores[-1][0, :, :, 0] = self.I
            cores[-1][1:1+n_pair, :, :, 0] = self.M
            cores[-1][1+n_pair, :, :, 0] = self.S
            if self.periodic:
                cores[-1][2+n_pair:2+2*n_pair, :, :, 0] = self.L.transpose([2, 0, 1])
        
            # Intermediate supercores
            for i in range(1, n_site - 1):
                cores[i] = np.zeros([n_size, n_dim, n_dim, n_size])
                cores[i][0, :, :, 0] = self.I
                cores[i][1:1+n_pair, :, :, 0] = self.M
                cores[i][1+n_pair, :, :, 0] = self.S
                cores[i][1+n_pair, :, :, 1:1+n_pair] = self.L
                cores[i][1+n_pair, :, :, 1+n_pair] = self.I
                if self.periodic:
                    for j in range(n_pair):
                        cores[i][2+n_pair+j, :, :, 2+n_pair+j] = self.I

        else:

            # reshape L and M into two 3D arrays if necessary (i.e., if only one two-site interaction)
            for i in range(n_site):
                if self.L[i] is not None:
                    if len(self.L[i].shape) == 2:
                        self.L[i] = self.L[i][:,:,None]
                if self.M[i] is not None:
                    if len(self.M[i].shape) == 2:
                        self.M[i] = self.M[i][None, :, :]

            # get core sizes
            n_pair, n_size = self.core_size()

            # First supercore
            cores[0] = np.zeros([1, n_dim, n_dim, n_size[0]])        #  row vector
            cores[0][0, :, :, 0] = self.S[0]
            cores[0][0, :, :, 1:1+n_pair[0]] = self.L[0]
            cores[0][0, :, :, 1+n_pair[0]] = self.I[0]
            if self.periodic:
                cores[0][0, :, :, 2+n_pair[0]:2+n_pair[0]+n_pair[-1]] = self.M[0].transpose([1, 2, 0])
            
            # Last supercore
            cores[-1] = np.zeros([n_size[-2], n_dim, n_dim, 1])       # column vector
            cores[-1][0, :, :, 0] = self.I[-1]
            cores[-1][1:1+n_pair[-2], :, :, 0] = self.M[-1]
            cores[-1][1+n_pair[-2], :, :, 0] = self.S[-1]
            if self.periodic:
                cores[-1][2+n_pair[-2]:2+n_pair[-2]+n_pair[-1], :, :, 0] = self.L[-1].transpose([2, 0, 1])
        
            # Intermediate supercores
            for i in range(1, n_site - 1):
                cores[i] = np.zeros([n_size[i-1], n_dim, n_dim, n_size[i]])
                cores[i][0, :, :, 0] = self.I[i]
                cores[i][1:1+n_pair[i-1], :, :, 0] = self.M[i]
                cores[i][1+n_pair[i-1], :, :, 0] = self.S[i]
                cores[i][1+n_pair[i-1], :, :, 1:1+n_pair[i]] = self.L[i]
                cores[i][1+n_pair[i-1], :, :, 1+n_pair[i]] = self.I[i]
                if self.periodic:
                    for j in range(n_pair[-1]):
                        cores[i][2+n_pair[i-1]+j, :, :, 2+n_pair[i]+j] = self.I[i]
        
        # Construct tensor train representation of the Hamiltonian
        hamilt = TT(cores)
        print("""
-----------------------------------
Setting up Hamiltonian in TT format 
-----------------------------------
        {hamilt} 
        """.format(hamilt=hamilt)) 

        # Qptionally use quantize TT representation: decompose 2^n tensor into
        # n 2-tensors makes computations much faster, but perhaps a little less
        # accurate. This is assuming that n_dim is an integer power of two
        if qtt:
            self.l2_dim = self.int_log2(n_dim)
            hamilt = TT.tt2qtt(hamilt, n_site  * [[2] * self.l2_dim] , n_site * [[2] * self.l2_dim] , threshold=1e-12)
            print("""
----------------------------------------
Converting the Hamiltonian to QTT format
----------------------------------------
            {hamilt} 
            """.format(hamilt=hamilt))
        else:
            self.l2_dim = 0

        self.represent = hamilt    

    def core_size(self):
        """
        Get number(s) of pair (L-M) interactions and core size(s)
        """

        if self.really_homogen:

            n_pair = self.L.shape[2]
            n_size = 2 + n_pair + self.periodic*n_pair

        else:

            n_pair = [0] * self.n_site
            n_size = [1] * self.n_site

            for i in range(self.n_site-1):
                n_pair[i] = self.L[i].shape[2]
                n_size[i] = 2 + n_pair[i] 
                if self.periodic:
                    n_size[i] = n_size[i] + self.M[0].shape[0]

            if self.periodic:
                n_pair[-1] = self.M[0].shape[0]

        return n_pair, n_size
    
    @staticmethod
    def int_log2(inp):
        """
        Get dual logarithm and check whether the result is integer, i.e. input must be an integer power of two
        """

        out = int(round(np.log2(inp)))
        
        if 2**out != inp:
            sys.exit ("Input must be an integer power of two")
        
        return out


