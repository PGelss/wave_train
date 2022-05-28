import numpy as np
from wave_train.dynamics.mechanics import Mechanics
from scikit_tt.tensor_train import residual_error

# Tolerance for suppressing imaginary part of (supposedly real!) expectation values
TOLERANCE = 1e6  # in units of the machine epsilon for doubles

class QuantumMechanics(Mechanics):
    """
    For quant.-mech. equations of motion,
    this class provides gateway functions
    for calculating and saving observables.
    """
    def __init__(self, hamilton):
        # super().__init__()  # TODO: Really needed ?!?!?

        # Hamiltonian function/operator
        self.hamilton = hamilton

        # Simulation titles (headers of plots) and timing
        self.head = ["" for x in range(self.num_steps+1)]
        self.cput = np.zeros(self.num_steps + 1)  # CPU time
        self.time = np.zeros(self.num_steps + 1)  # simulated time

        # All state vectors (only for quasi-exact solvers, where chains not too long)
        if self.solver == 'qe':
            self.psi_all = np.zeros((self.num_steps+1, self.hamilton.n_dim**self.hamilton.n_site),dtype=complex)

        # Elementary observables
        self.nrgy = np.zeros(self.num_steps + 1)  # total energy
        self.norm = np.zeros(self.num_steps + 1)  # norm of state vector
        self.auto = np.zeros(self.num_steps + 1, dtype=complex)  # autocorrelation
        if self.name == 'TISE':
            self.iter = np.zeros(self.num_steps + 1)  # number of iterations
            self.nres = np.zeros(self.num_steps + 1)  # norm of residue

        # More observables (from reduced density matrices)
        self.rho_site = np.zeros((self.num_steps + 1, self.hamilton.n_site,
                             self.hamilton.n_dim, self.hamilton.n_dim), dtype=complex)
        self.position = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # position
        self.momentum = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # momentum
        self.pos_squa = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # position squared
        self.mom_squa = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # momentum uncertainty
        self.pos_wide = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # position uncertainty
        self.mom_wide = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # momentum uncertainty
        self.unc_prod = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # uncertainty product

        if self.hamilton.bipartite:
            self.ex_numbr = np.zeros((self.num_steps + 1, self.hamilton.n_site)) # quantum number for excitons
            self.ph_numbr = np.zeros((self.num_steps + 1, self.hamilton.n_site)) # quantum number for phonons
        else:
            self.qu_numbr = np.zeros((self.num_steps + 1, self.hamilton.n_site)) # quantum number

    def observe(self, i, iterations=0):
        """
        Update and print expectation values of observables
        from quantum-mechanical simulations (TISE or TDSE)
        ----------------------------

        Parameters:
            i: int
                index of time step (TDSE) or index of stationary state (TISE)
            iterations: int
                number of ALS iterations needed (TISE only)
        """

        # Simulation titles (headers of plots)
        self.head[i] = self.title
        self.cput[i] = self.cpu_time
        self.time[i] = i * self.step_size

        # All state vecors (only for quasi-exact solvers, where chains not too long)
        if self.solver == 'qe':
            self.psi_all[i,:] = self.psi

        # Elementary observables (directly from psi)
        if self.solver == 'qe':
            psi_trans = self.psi.conj().T
            self.nrgy[i] = np.real_if_close(psi_trans @ (self.operator @ self.psi))
            self.norm[i] = np.real_if_close(psi_trans @ self.psi)
            self.nrgy[i] /= self.norm[i]
            self.auto[i] = self.psi_0.conj().T @ self.psi

        else:
            self.nrgy[i] = np.real_if_close(self.expect(self.hamilton.represent, self.psi,
                                                        self.hamilton.n_site, self.hamilton.l2_dim),
                                            tol=TOLERANCE)
            self.norm[i] = np.real_if_close(self.bra_ket(self.psi, self.psi, self.hamilton.n_site, self.hamilton.l2_dim),
                                            tol=TOLERANCE) ** ( 1 / 2)  # = self.psi_0.norm(p=2)
            self.auto[i] = self.bra_ket(self.psi_0, self.psi, self.hamilton.n_site, self.hamilton.l2_dim)
        if self.name == 'TISE':
            if self.conv_eps == -1:
                self.iter[i] = self.repeats
            else:
                self.iter[i] = iterations
            if self.solver == 'qe':
                self.nres[i] = np.real_if_close(np.linalg.norm(self.operator @ self.psi - self.nrgy[i] * self.psi),
                                                tol=TOLERANCE)
            else:
                self.nres[i] = np.real_if_close(self.residue(self.hamilton.represent, self.psi, self.nrgy[i],
                                                             self.hamilton.n_site, self.hamilton.l2_dim),
                                                tol=TOLERANCE)

        # Console output
        print(63 * '-')
        print(self.head[i])
        print(63 * '-')
        print(' ')

        if self.name == 'TISE':
            print('After iterations : ' + str(self.iter[i]))
        print('Total energy     : ' + str(self.nrgy[i]))
        if hasattr(self, 'exct'):
            print('Exact energy     : ' + str(self.exct[i]))
            print('Difference       : ' + str(self.nrgy[i] - self.exct[i]))
        print('Norm of psi      : ' + str(self.norm[i]))
        print('Autocorrelation  : ' + str(self.auto[i]))
        if self.name == 'TISE':
            print('Norm of residue  : ' + str(self.nres[i]))


        if self.solver != 'qe':

            print('Rank of psi (TT) : ' + str(self.psi.ranks))

            # Reduced density matrices
            self.rho_site[i] = self.reduce(self.psi, self.hamilton.n_site, self.hamilton.l2_dim)

            # Header of table with site-specific information
            print(' ')
            if self.hamilton.bipartite:
                print('site |  qu_n (ex) |  qu_n (ph) |   position |   momentum |    Dx (ph) |    Dp (ph) |  DxDp (ph)')
                print(96 * '-')
            elif self.hamilton.classical:
                print('site | qu_number |   position |   momentum |    Dx (ph) |    Dp (ph) |  DxDp (ph)')
                print(82 * '-')
            else:
                print(
                    'site |  qu_number')
                print(17 * '-')

            # Entries of table with site-specific information
            for j in range(self.hamilton.n_site):
                rho = self.rho_site[i,j]
                if self.hamilton.bipartite:
                    self.ex_numbr[i, j] = np.real_if_close(                               np.trace(rho @ self.hamilton.ex_numbr), tol=TOLERANCE)
                    self.ph_numbr[i, j] = np.real_if_close(                               np.trace(rho @ self.hamilton.ph_numbr), tol=TOLERANCE)
                else:
                    self.qu_numbr[i, j] = np.real_if_close(                               np.trace(rho @ self.hamilton.qu_numbr), tol=TOLERANCE)
                if self.hamilton.classical:
                    self.position[i, j] =     np.real_if_close(self.hamilton.pos_conv[j]    * np.trace(rho @ self.hamilton.position), tol=TOLERANCE)
                    self.momentum[i, j] =     np.real_if_close(self.hamilton.mom_conv[j]    * np.trace(rho @ self.hamilton.momentum), tol=TOLERANCE)
                    self.pos_squa[i, j] =     np.real_if_close(self.hamilton.pos_conv[j]**2 * np.trace(rho @ self.hamilton.pos_squa), tol=TOLERANCE)
                    self.mom_squa[i, j] =     np.real_if_close(self.hamilton.mom_conv[j]**2 * np.trace(rho @ self.hamilton.mom_squa), tol=TOLERANCE)
                    self.pos_wide[i, j] =     np.real_if_close(                  np.sqrt(self.pos_squa[i,j] - self.position[i,j]**2), tol=TOLERANCE)
                    self.mom_wide[i, j] =     np.real_if_close(                  np.sqrt(self.mom_squa[i,j] - self.momentum[i,j]**2), tol=TOLERANCE)
                    self.unc_prod[i, j] =                                             self.pos_wide[i,j] * self.mom_wide[i,j]  # uncertainty product

                if self.hamilton.bipartite:
                    print(str("%4d" % j) + ' | ' + str("%10f" % self.ex_numbr[i, j]) + ' | ' + str(
                        "%10f" % self.ph_numbr[i, j]) + ' | ' + str("%10f" % self.position[i, j]) + ' | ' + str(
                        "%10f" % self.momentum[i, j]) + ' | ' + str("%10f" % self.pos_wide[i, j]) + " | " + \
                          str("%10f" % self.mom_wide[i, j]) + " | " + str("%10f") % self.unc_prod[i, j] )
                elif self.hamilton.classical:
                    print(str("%4d" % j) + ' |' + str("%10f" % self.qu_numbr[i, j]) + ' | ' + str(
                        "%10f" % self.position[i, j]) + ' | ' + str("%10f" % self.momentum[i, j]) + ' | ' + \
                          str("%10f" % self.pos_wide[i, j]) + ' | ' + str("%10f" % self.mom_wide[i, j]) + " | " + \
                          str("%10f" % self.unc_prod[i, j]))
                else:
                    print(str("%4d" % j) + ' |' + str("%10f" % self.qu_numbr[i, j]))

            # Footer of table with site-specific information
            if self.hamilton.bipartite:
                print(96 * '-')
                print(' sum | ' + str("%10f" % np.sum(self.ex_numbr[i, :])) + ' | ' + str(
                    "%10f" % np.sum(self.ph_numbr[i, :])))
            elif self.hamilton.classical:
                print(82 * '-')
                print(' sum |' + str("%10f" % np.sum(self.qu_numbr[i, :])))
            else:
                print(17 * '-')
                print(' sum |' + str("%10f" % np.sum(self.qu_numbr[i, :])))
            print(' ')

        # RMSD of positions from reference solution (if available)
        if hasattr(self,'ref_pos'):
            pos_now = self.position[i, :]
            pos_ref = self.ref_pos[i, :]
            self.rmsd[i] = np.linalg.norm(pos_now - pos_ref) / \
                           np.sqrt(self.hamilton.n_site)
            print('RMSD of positions : ' + str(self.rmsd[i]))

        # RMSD of populations of quantum states from reference solution (if available)
        if hasattr(self, 'ref_rho'):
            msd = 0 # mean squared deviation
            for j in range(self.hamilton.n_site):
                pop_now = np.real_if_close(np.diag(self.rho_site [i, j]))  # vector of length n_dim
                pop_ref = np.real_if_close(np.diag(self.ref_rho [i, j]))   # vector of length n_dim
                msd += np.sum((pop_now - pop_ref)**2)                      # sum over elements of a vector
            self.rmsd[i] = np.sqrt(msd / (self.hamilton.n_site * self.hamilton.n_dim))
            print('RMSD of populations : ' + str(self.rmsd[i]))

        # RMSD of quantum state vectors from reference solution (if available)
        if hasattr(self, 'ref_psi'):
            psi_now = (self.psi.matricize()).astype(complex)
            psi_ref = self.ref_psi[i, :]
            self.rmsd[i] = np.linalg.norm(psi_now - psi_ref) / \
                           np.sqrt(self.hamilton.n_dim**self.hamilton.n_site)
            print('RMSD of state vectors : ' + str(self.rmsd[i]))

        print (' ')

    @staticmethod
    def bra_ket(bra, ket, n_site, l2_dim=0):
        """
        Get inner/scalar product <bra|ket> = integral/sum over conj(phi)*psi

        Parameters
        ----------
        bra, ket : two instances of TT class of identical order
            quantum state vectors; the first (bra) will cc'ed
        n_site : integer
            length of chain
        l2_dim : integer (optional)
            dual logarithm

        Returns
        --------

        output: float
            scalar product of two quantum state
        """

        if l2_dim > 0:
            bra = bra.qtt2tt([l2_dim] * n_site)
            ket = ket.qtt2tt([l2_dim] * n_site)

        bra_trans = bra.transpose(conjugate=True)

        output = bra_trans @ ket

        return output

    @staticmethod
    def residue(opr, psi, ene, n_site, l2_dim=0):
        """
        Residue of an eigenproblem  | op|psi> - E|psi> |

        Parameters
        ----------
        opr : instance of TT class
            quantum operator
        psi : instance of TT class
            quantum state vector
        ene : float
            energy
        n_site : integer
            length of chain
        l2_dim : integer (optional)
            dual logarithm

        Returns
        -------
        output : float
            norm of residue
        """

        if l2_dim > 0:
            opr = opr.qtt2tt([l2_dim] * n_site)
            psi = psi.qtt2tt([l2_dim] * n_site)

        output = residual_error(opr, psi, ene * psi)
        # output = (opr@psi-ene*psi).norm()

        return output

    @staticmethod
    def expect(opr, psi, n_site, l2_dim=0):
        """
        Get expectation value <psi|op|psi>/<psi|psi> = integral/sum over conj(psi)*Op(psi)

        Parameters
        ----------
        opr : instance of TT class
            quantum operator
        psi : instance of TT class
            quantum state vector
        n_site : integer
            length of chain
        l2_dim : integer (optional)
            dual logarithm

        Returns
        -------
        output : float
            expectation value of an operator wrt to a quantum state
        """

        if l2_dim > 0:
            opr = opr.qtt2tt([l2_dim] * n_site)
            psi = psi.qtt2tt([l2_dim] * n_site)

        psi_trans = psi.transpose(conjugate=True)
        output = psi_trans @ (opr @ psi)

        output /= (psi_trans @ psi)

        return output

    @staticmethod
    def reduce(psi, n_site, l2_dim=0):
        """
        Get reduced density matrices from quantum state vector psi

        Parameters
        ----------
        psi : instance of TT class
            quantum state vector in tensor train format
        n_site : integer
            length of chain
        l2_dim : integer (optional)
            dual logarithm

        Returns
        -------
        rhos : matrices
            density matrices for the d-th degree of freedom
        """

        rhos = []

        if l2_dim > 0:
            psi = psi.qtt2tt([l2_dim] * n_site)

        for d in range(psi.order):

            psi_tmp = psi.transpose(cores=[i for i in range(d)]+[i for i in range(d+1, psi.order)])
            psi_trans = psi_tmp.transpose(conjugate=True)
            rho = (psi_tmp @ psi_trans).matricize()
            rhos.append(rho)

        return rhos