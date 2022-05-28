import numpy as np
from wave_train.dynamics.mechanics import Mechanics

# Tolerance for suppressing imaginary part of (supposedly real!) expectation values
TOLERANCE = 1e6  # in units of the machine epsilon for doubles

class QuantClassMechanics(Mechanics):
    """
    For quant.-class. dynamics simulations,
    this class provides gateway functions
    for calculating and saving system observables.
    Moreover, there are propagators for numerical
    integration of quant.-class. equations of motion
    """
    def __init__(self, hamilton):
        # super().__init__()  # TODO: Really needed ?!?!?

        # Hamiltonian function/operator
        self.hamilton = hamilton

        # Simulation titles (headers of plots) and timing
        self.head = ["" for x in range(self.num_steps+1)]
        self.cput = np.zeros(self.num_steps + 1)  # CPU time
        self.time = np.zeros(self.num_steps + 1)  # simulated time

        # Three forms of energies
        self.e_quant = np.zeros(self.num_steps + 1)  # energy of quantum subsystem
        self.e_qu_cl = np.zeros(self.num_steps + 1)  # energy of quantum-classical coupling
        self.e_class = np.zeros(self.num_steps + 1)  # energy of classical subsystem
        self.nrgy = np.zeros(self.num_steps + 1)  # total energy: quant + class

        # Quantum subsystem
        self.norm = np.zeros(self.num_steps + 1)  # norm of state vector
        self.auto = np.zeros(self.num_steps + 1, dtype=complex) # autocorrelation
        self.ex_numbr = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # quantum number for excitons

        # Classical subsystem
        self.position = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # position
        self.momentum = np.zeros((self.num_steps + 1, self.hamilton.n_site))  # momentum
        self.potential = np.zeros(self.num_steps + 1)  # potential energy
        self.kinetic = np.zeros(self.num_steps + 1)  # kinetic energy

    def observe(self, i, iterations=0):
        """
        Update and print expectation values of observables
        from quantum-classical simulations (QCMD)
        --------------------------------------------------

        Parameters:
            i: int
                index of time step (TDSE) or index of stationary state (TISE)
            iterations: int
                number of ALS iterations needed (TISE only)
        """

        # Simulation titles (headers of plots) and timing
        self.head[i] = self.title
        self.cput[i] = self.cpu_time
        self.time[i] = i * self.step_size

        # Three forms of energies
        self.e_quant[i] = np.real_if_close(np.conjugate(self.psi) @ (self.hamilt_quant @ self.psi))  # H is a matrix
        self.e_qu_cl[i] = np.real_if_close(np.conjugate(self.psi) @ (self.hamilt_qu_cl * self.psi))  # H is a vector
        self.e_class[i] = self.hamilton.ph.potential(self.pos) + \
                          self.hamilton.ph.kinetic(self.mom)
        self.nrgy[i] = self.e_quant[i] + self.e_qu_cl[i] + self.e_class[i]

        # Quantum subsystem
        self.norm[i] = np.real_if_close(np.conjugate(self.psi) @ self.psi)
        self.auto[i] = np.real_if_close(np.conjugate(self.psi_0) @ self.psi)

        # Console output
        print(50 * '-')
        print(self.head[i])
        print(50 * '-')
        print(' ')
        print('Energy (quant.)  : ' + str(self.e_quant[i]))
        print('Energy (qu.-cl.) : ' + str(self.e_qu_cl[i]))
        print('Energy (class.)  : ' + str(self.e_class[i]))
        print('Total energy     : ' + str(self.nrgy[i]))
        print('Norm of psi      : ' + str(self.norm[i]))
        print('Autocorrelation  : ' + str(self.auto[i]))

        # Header of table with site-specific information
        print(' ')
        if self.hamilton.bipartite:
            print('site |  density   |   position |   momentum ')
            print(43 * '-')

        # Entries of table with site-specific information
        for j in range(self.hamilton.n_site):
            self.position[i, j] = self.pos[j]
            self.momentum[i, j] = self.mom[j]
            self.ex_numbr[i, j] = np.abs(self.psi[j]) ** 2

            print(str("%4d" % j) + ' | ' + str("%10f" % self.ex_numbr[i, j]) + ' | ' + str("%10f" % self.position[i, j]) + ' | ' + str(
                "%10f" % self.momentum[i, j])  )


        # Footer of table with site-specific information
        print(43 * '-')
        print(' sum' +
              ' | ' + str("%10f" % np.sum(self.ex_numbr[i, :])) +
              ' | ' + str("%10f" % np.sum(self.position[i, :])) +
              ' | ' + str("%10f" % np.sum(self.momentum[i, :])) )
        print (' ')

        # RMSD of positions from reference solution (if available)
        if hasattr(self,'ref_pos'):
            pos_now = self.position[i, :]
            pos_ref = self.ref_pos[i, :]
            self.rmsd[i] = np.linalg.norm(pos_now - pos_ref) / \
                           np.sqrt(self.hamilton.n_site)
            print('RMSD of positions : ' + str(self.rmsd[i]))
            print(' ')
