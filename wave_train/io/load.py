import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

class Load:
    """
    Loading all data from a previous simulation from a pic[kle]-file
    """

    def __init__(self, file_name, file_type):
        """
         file_name: string
         file_type: string
            Has to be a pickle-file
        """
        self.file_name = file_name
        self.file_type = file_type
        self.data_file = file_name + '.' + file_type

        # Data file has to be a pickle-format
        if self.file_type not in ['pic', 'pickle']:
            print ("Filename extension indicates a not supported input format. Data cannot be loaded.")
            sys.exit()

        # Load simulation data from file
        try:
            with open(self.data_file, 'rb') as dynamics_file:  # 'rb' stands for "read binary"
                dynamics = pickle.load(dynamics_file)
        except pickle.PickleError:
            sys.exit("Unpickling process failed")

        # Create dictionary representation of class
        dict_repr = vars(dynamics)
        for key, value in dict_repr.items():
            # set attributes of this class with values loaded from pickle file
            setattr(self, key, value)

    def __str__(self):
        info = """
---------------------------------------------------------------
Loading previous simulation data 
from data file : {}
---------------------------------------------------------------

Name of equation solved       : {}
Name of Hamiltonian used      : {}

Number of sites in chain/ring : {}
Periodic boundary conditions  : {}
Homogeneous chain/ring        : {}
        """.format(self.data_file, self.name, self.hamilton.name,
                   self.hamilton.n_site, self.hamilton.periodic, self.hamilton.homogen)

        return info

    # Empty method, but needed for visualization
    def start_solve(self):
        print(self)

    # Essentially empty method, but needed for visualization
    def update_solve(self, i):
        print(63*'-')
        print('step/state', i)
        print(63*'-')
        print(' ')
        print('total energy      : ' + str(self.nrgy[i]))
        print(' ')

    # Check conservation of energy, norm, and RMSD (if available) in TDSE simulations
    def check_ENR (self):

        # Console output
        print(self)
        if hasattr(self, 'threshold'):
            print('Threshold for reduced SVD        : ', str(self.threshold))
        if hasattr(self, 'max_rank'):
            print('Maximum rank of solution         : ', str(self.max_rank))
        if hasattr(self, 'repeats'):
            print('Maximum number of ALS iterations : ', str(self.repeats))

        # Create a new figure with title and set its size
        plt.figure('check_ENR', figsize=(5, 7))  # Width, height in inches.

        # Energy versus time
        if hasattr(self, 'nrgy'):
            print(' ')
            print('******************')
            print('Energy versus time')
            print('******************')
            print(' ')
            print('Slope     : ' + str(self.nrgy_slope))
            print('Intercept : ' + str(self.nrgy_inter))
            print('RMSD      : ' + str(self.nrgy_rmsd))
            plt.subplot(3, 1, 1)
            plt.plot(self.time, self.nrgy, 'o')
            plt.plot(self.time, self.nrgy_slope * self.time + self.nrgy_inter)
            plt.plot(self.time, [self.nrgy[0] + self.nrgy_rmsd] * len(self.time), '-')
            plt.plot(self.time, [self.nrgy[0] - self.nrgy_rmsd] * len(self.time), '-')
            plt.ylabel('energy')

        # Norm versus time
        if hasattr(self, 'norm'):
            print(' ')
            print('****************')
            print('Norm versus time')
            print('****************')
            print(' ')
            print('Slope     : ' + str(self.norm_slope))
            print('Intercept : ' + str(self.norm_inter))
            print('RMSD      : ' + str(self.norm_rmsd))
            plt.subplot(3, 1, 2)
            plt.plot(self.time, self.norm, 'o')
            plt.plot(self.time, self.norm_slope * self.time + self.norm_inter)
            plt.plot(self.time, [self.norm[0] + self.norm_rmsd] * len(self.time), '-')
            plt.plot(self.time, [self.norm[0] - self.norm_rmsd] * len(self.time), '-')
            plt.ylabel('norm')

        # RMSD versus time
        if hasattr(self, 'rmsd'):
            print(' ')
            print('****************')
            print('RMSD versus time')
            print('****************')
            print(' ')
            print('Slope     : ' + str(self.rmsd_slope))
            print('Intercept : ' + str(self.rmsd_inter))
            print('RMSD      : ' + str(self.rmsd_rmsd))
            plt.subplot(3, 1, 3)
            plt.plot(self.time, self.rmsd, 'o')
            plt.plot(self.time, self.rmsd_slope * self.time + self.rmsd_inter)
            plt.plot(self.time, [self.rmsd[0] + self.rmsd_rmsd] * len(self.time), '-')
            plt.ylabel('RMSD')
            plt.xlabel('time')

        # Adding a "super title"
        plt.suptitle(self.file_name)

        # Save graphics as a PDF file
        pdf_file = self.file_name + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file)
        plt.savefig(pdf_file)

        # Show graphics on screen
        plt.show()

    # Redistribution of energy between kinetic and potential in CEoM simulations
    def check_EKP (self):

        # Console output
        print(self)

        # Create a new figure with title and set its size
        plt.figure('check_EKP', figsize=(5, 7))  # Width, height in inches.

        # Various forms of energy versus time
        if hasattr(self, 'nrgy'):
            plt.plot(self.time, self.nrgy, 'o-', label='total')

        if hasattr(self,'kinetic'):
            plt.plot(self.time, self.kinetic, 'o-', label='kinetic')

        if hasattr(self,'potential'):
            plt.plot(self.time, self.potential, 'o-', label='potential')

        # Plot labels and legend
        plt.ylabel('energy')
        plt.xlabel('time')
        plt.legend(loc='best')

        # Adding a "super title"
        file_name = self.data_file.split('.')[0]
        plt.suptitle(file_name)

        # Save graphics as a PDF file
        pdf_file = file_name + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file)
        plt.savefig(pdf_file)

        # Show graphics on screen
        plt.show()

    # Redistribution of energy between quantum and classical subsystem in QCMD simulations
    def check_EQC (self):

        # Console output
        print(self)

        # Create a new figure with title and set its size
        plt.figure('check_EQC', figsize=(5, 7))  # Width, height in inches.

        # Various forms of energy versus time
        if hasattr(self, 'nrgy'):
            plt.plot(self.time, self.nrgy, 'o-', label='total')

        if hasattr(self,'e_quant'):
            plt.plot(self.time, self.e_quant, 'o-', label='quant')

        if hasattr(self,'e_class'):
            plt.plot(self.time, self.e_class, 'o-', label='class')

        if hasattr(self,'e_qu_cl'):
            plt.plot(self.time, self.e_qu_cl, 'o-', label='qu_cl')

        # Plot labels and legend
        plt.ylabel('energy')
        plt.xlabel('time')
        plt.legend(loc='best')

        # Adding a "super title"
        file_name = self.data_file.split('.')[0]
        plt.suptitle(file_name)

        # Save graphics as a PDF file
        pdf_file = file_name + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file)
        plt.savefig(pdf_file)

        # Show graphics on screen
        plt.show()

    # Check populations of quantum states
    def check_pop(self):

        for i in range(self.num_steps+1):
            print(' ')
            print(42*'*')
            print('time step : ' + str(i))
            print(42*'*')
            for j in range(self.hamilton.n_site):
                pops = np.real_if_close(np.diag(self.rho_site[i, j]))
                print (str(j) + ' : ' + str(pops))



