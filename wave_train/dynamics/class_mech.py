import numpy as np
from wave_train.dynamics.mechanics import Mechanics

class ClassicalMechanics(Mechanics):
    """
    For class. mech. dynamics simulations,
    this class provides gateway functions
    for calculating and saving system observables.
    Moreover, there are propagators for numerical
    integration of class.-mech. equations of motion
    """
    def __init__(self, hamilton):
        # super().__init__()  # TODO: Really needed ?!?!?

        # Hamiltonian function
        self.hamilton = hamilton

        # Simulation titles (headers of plots) and timing
        self.head = ["" for x in range(self.num_steps + 1)]
        self.cput = np.zeros(self.num_steps + 1) # CPU time
        self.time = np.zeros(self.num_steps + 1) # simulated time

        # Elementary observables
        self.nrgy = np.zeros(self.num_steps + 1) # total energy
        self.potential = np.zeros(self.num_steps + 1) # potential energy
        self.kinetic   = np.zeros(self.num_steps + 1) # kinetic energy
        self.position  = np.zeros((self.num_steps + 1, self.hamilton.n_site)) # position
        self.momentum  = np.zeros((self.num_steps + 1, self.hamilton.n_site)) # momentum

    def observe(self, i):
        """
        Update and print observables
        from classical simulations (CEoM)
        ---------------------------------

        Parameters:
            i: int
                index of time step
        """

        # Simulation titles (headers of plots)
        self.head[i] = self.title
        self.cput[i] = self.cpu_time
        self.time[i] = i * self.step_size

        # Classical observables
        self.potential[i]  = self.hamilton.potential (self.pos)
        self.kinetic[i]    = self.hamilton.kinetic   (self.mom)
        self.nrgy[i] = self.potential[i] + self.kinetic[i]
        self.position[i,:] = self.pos
        self.momentum[i,:] = self.mom

        # Console output
        print(63*'-')
        print(self.head[i])
        print(63*'-')
        print(' ')
        print('potential energy : ' + str(self.potential[i]))
        print('kinetic energy   : ' + str(self.kinetic[i]))
        print('total energy     : ' + str(self.nrgy[i]))
        print(' ')
        print('site |   position |   momentum')
        print(30 * '-')
        for j in range(self.hamilton.n_site):
            print(str("%4d" % j) + ' | ' + str("%10f" % self.position[i, j]) + ' | ' + str("%10f" % self.momentum[i, j]))
        print(30 * '-')
        print(' ')




