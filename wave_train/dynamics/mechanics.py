import sys
import pickle
import numpy as np
import scipy.io as sio
from scipy import stats
from datetime import datetime

class Mechanics:
    # def __init__(self):         # TODO: Really needed ?!?!?
        # self.compare = None
        # self.load_file = None
        # self.save_file = None
        # self.num_steps = 0
        # self.solver = None
        # self.name = None

    def save(self):
        """
        Save expectation values of observables
        and related staff to data files:
            Currently implemented: mat-files, pickle files
        """

        # Current date and time formatted as: dd/mm/YY H:M:S
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        if self.save_file is not None:

            print('-----------------------------')
            print('Saving generated data to file')
            print('-----------------------------')
            print(' ')
            print('Current date and time : ' + dt_string)
            print('Data file name        : ' + self.save_file)
            print(' ')

            # Choice of filename extension
            file_suffix = self.save_file.split('.')[-1]
            if file_suffix == 'mat':
                dict_for_mat = self.__getstate__()
                sio.savemat(self.save_file, dict_for_mat, appendmat=False)
            elif file_suffix in ['pic','pickle']:
                pickle.dump(self, open(self.save_file, 'wb')) # 'wb' stands for "write binary"
            else:
                sys.exit('Filename extension indicates a non-supported output format. Data cannot be saved.')

        else:
            print('-------------------------------------')
            print('Terminate without saving data to file')
            print('-------------------------------------')
            print(' ')
            print('Current date and time : ' + dt_string)
            print(' ')

    def load(self):
        """
        If available, load reference solution from data file
            Currently implemented: pickle files only, but mat-files shouldn't be too difficult either
        """

        if self.load_file is not None:

            print('--------------------------------')
            print('Loading reference data from file')
            print('--------------------------------')
            print(' ')
            print('Loading from data file        : ', self.load_file)
            print(' ')

            # Check filename extension
            file_suffix = self.load_file.split('.')[-1]
            if file_suffix not in ['pic','pickle']:
                sys.exit('Filename extension indicates a non-supported output format. Data cannot be loaded.')

            # Open pickle file and load reference data
            with open(self.load_file, 'rb') as reference_file:  # 'rb' stands for "read binary"
                reference = pickle.load(reference_file)

            # Convert dictionary representation into class attributes
            dict_repr = vars(reference)
            for key, value in dict_repr.items():
                # set attributes of this class with values loaded from pickle file
                setattr(reference, key, value)
                # print(key, ' : ', value)

            # Checks for consistency of reference solution
            print('Number of sites in chain/ring : ' + str(reference.hamilton.n_site))
            if reference.hamilton.n_site != self.hamilton.n_site:
                sys.exit('Inconsistent (n_site) with current simulation')

            print('Periodic boundary conditions  : ' + str(reference.hamilton.periodic))
            if reference.hamilton.periodic != self.hamilton.periodic:
                sys.exit('Inconsistent (periodic) with current simulation')

            print('Homogeneous chain/ring        : ' + str(reference.hamilton.homogen))
            if reference.hamilton.homogen != self.hamilton.homogen:
                sys.exit('Inconsistent (homogen) with current simulation')

            print('Number of (main) time steps   : ' + str(reference.num_steps))
            if reference.num_steps < self.num_steps:
                sys.exit('Inconsistent (num_steps) with current simulation')

            # Root mean squared deviation of solution from reference
            self.rmsd = np.zeros(self.num_steps + 1)  # root mean squared displacement from reference solution

            # Extract quantities of interest from reference solutions: 3 choices
            if self.compare == 'pos':  # positions (quantum or classical)
                self.ref_pos = reference.position  # position vectors
                print('Extracting positions          : shape = ' + str(self.ref_pos.shape))

            elif self.compare == 'pop':  # populations of quantum states
                self.ref_rho = reference.rho_site  # (reduced) density matrices
                print('Extracting densities          : shape = ' + str(self.ref_rho.shape))

            elif self.compare == 'psi':  # quantum state vectors
                self.ref_psi = reference.psi_all
                print('Extracting state vectors      : shape = ' + str(self.ref_psi.shape))

            else:
                sys.exit('Wrong choice of how to compare with reference data')

            print (' ')


    def __getstate__(self):
        """
        Controls the output of the pickle serialization operation.
        The following attributes are removed from the pickling process
        in order to keep the file sizes manageable

        TDSE: psi_guess, psi_0, represent
        TISE: psi_guess, psi_0, previous, represent
        """
        processed_dict = self.__dict__.copy()
        for key in ['psi_guess', 'psi_0', 'previous', 'represent']:
            # if key is not present in dictionary, None is returned
            processed_dict.pop(key, None)
        return processed_dict

    def linear_regression(self):
        """
        Linear regression of total energy and norm and RMSD versus time
            Get drift and variance and what not
            Only for TDSE and CEoM, not for TISE
        """

        # Linear regression and RMSD of total energy versus time
        if hasattr(self, 'nrgy'):
            self.nrgy_slope, self.nrgy_inter, self.nrgy_r_val, self.nrgy_p_val, self.nrgy_error = stats.linregress(self.time, self.nrgy)
            self.nrgy_rmsd = np.linalg.norm(self.nrgy-self.nrgy[0])/np.sqrt(len(self.nrgy))  # Frobenius divided by number of time steps
            print ('------------------------------------------------------')
            print ('Linear regression and RMSD of total energy versus time')
            print ('------------------------------------------------------')
            print(' ')
            print ('Slope of the line       : ' + str(self.nrgy_slope))  # slope of the regression line
            print ('Intercept of the line   : ' + str(self.nrgy_inter))  # intercept of the regression line
            print ('Relative drift          : ' + str(self.nrgy_slope/self.nrgy_inter))
            print ('Correlation coefficient : ' + str(self.nrgy_r_val))  # correlation coefficient
            print ('Two-sided p-value       : ' + str(self.nrgy_p_val))  # two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
            print ('Standard error          : ' + str(self.nrgy_error))  # Standard error of the estimate
            print ('Relative error          : ' + str(self.nrgy_error/self.nrgy_inter))
            print ('RMSD from time zero     : ' + str(self.nrgy_rmsd))   # Root mean squared deviation
            print (' ')

        # Linear regression and RMSD of norm versus time
        if hasattr(self, 'norm'):
            self.norm_slope, self.norm_inter, self.norm_r_val, self.norm_p_val, self.norm_error = stats.linregress(self.time, self.norm)
            self.norm_rmsd = np.linalg.norm(self.norm-self.norm[0])/np.sqrt(len(self.norm))  # Frobenius divided by number of time steps
            print ('----------------------------------------------')
            print ('Linear regression and RMSD of norm versus time')
            print ('----------------------------------------------')
            print (' ')
            print('Slope of the line       : ' + str(self.norm_slope))  # slope of the regression line
            print('Intercept of the line   : ' + str(self.norm_inter))  # intercept of the regression line
            print('Relative drift          : ' + str(self.norm_slope / self.norm_inter))
            print('Correlation coefficient : ' + str(self.norm_r_val))  # correlation coefficient
            print('Two-sided p-value       : ' + str(self.norm_p_val))  # two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
            print('Standard error          : ' + str(self.norm_error))  # Standard error of the estimate
            print('Relative error          : ' + str(self.norm_error / self.norm_inter))
            print('RMSD from time zero     : ' + str(self.norm_rmsd))   # Root mean squared deviation
            print (' ')

        # Linear regression and RMSD of RMSD versus time
        if hasattr(self, 'rmsd'):
            self.rmsd_slope, self.rmsd_inter, self.rmsd_r_val, self.rmsd_p_val, self.rmsd_error = stats.linregress(self.time, self.rmsd)
            self.rmsd_rmsd = np.linalg.norm(self.rmsd-self.rmsd[0])/np.sqrt(len(self.rmsd))  # Frobenius divided by number of time steps
            print ('----------------------------------------------')
            print ('Linear regression and RMSD of RMSD versus time')
            print ('----------------------------------------------')
            print (' ')
            print('Slope of the line       : ' + str(self.rmsd_slope))  # slope of the regression line
            print('Intercept of the line   : ' + str(self.rmsd_inter))  # intercept of the regression line
            print('Correlation coefficient : ' + str(self.rmsd_r_val))  # correlation coefficient
            print('Two-sided p-value       : ' + str(self.rmsd_p_val))  # two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
            print('Standard error          : ' + str(self.rmsd_error))  # Standard error of the estimate
            print('RMSD from time zero     : ' + str(self.rmsd_rmsd))   # Root mean squared deviation
            print (' ')

    def gaussian(self, x_0=None, k_0=0, w_0=1):
        """
         Construct an initial quantum state as a Gaussian bell-shaped wavepacket
         This is a typical solution of a linear TDSE for a free particle
        """

        # If not specified otherwise, put wavepacket at|near center of chain
        if x_0 is None:
            x_0 = self.hamilton.n_site // 2

        print("------------------------------------------------------")
        print("Setting up an initial Gaussian bell-shaped wave packet")
        print("------------------------------------------------------")
        print(" ")
        print("mean position   : ", str(x_0))
        print("mean momentum   : ", str(k_0))
        print("width parameter : ", str(w_0))
        print(" ")
        n = self.hamilton.n_site
        x = np.arange(0, n)
        g = np.exp(-((x - x_0) / (2 * w_0)) ** 2)

        # Avoid imaginary parts of Gaussian unless necessary
        if k_0 != 0:
            g *= np.exp(-1j * k_0 * (x - x_0))

        # Truncate low values to keep things simple
        for k in range(n):
            if np.abs(g[k]) < 0.01:
                g[k] = 0
            print(k, g[k])
        print(" ")

        self.fundamental(g.tolist())

    def sec_hyp(self, x_0=None, k_0=0, w_0=1):
        """
          Construct an initial quantum state as a sech=1/cosh bell-shaped wavepacket
          This is a typical solution of a cubic non-linear TDSE for a soliton
          See, for example, Davydov's theory for solitons in helical proteins
         """

        # If not specified otherwise, put wavepacket at|near center of chain
        if x_0 is None:
            x_0 = self.hamilton.n_site // 2

        print("----------------------------------------------------------")
        print("Setting up an initial hyperbolic secant-shaped wave packet")
        print("----------------------------------------------------------")
        print(" ")
        print("mean position   : ", str(x_0))
        print("mean momentum   : ", str(k_0))
        print("width parameter : ", str(w_0))
        print(" ")
        n = self.hamilton.n_site
        x = np.arange(0, n)
        c = 1.0 / np.cosh((x - x_0) / (4 * w_0))

        # Avoid imaginary parts of Gaussian unless necessary
        if k_0 != 0:
            c *= np.exp(-1j * k_0 * (x - x_0))

        # Truncate low values to keep things simple
        for k in range(n):
            if np.abs(c[k]) < 0.01:
                c[k] = 0
            print(k, c[k])
        print(" ")

        self.fundamental(c.tolist())

