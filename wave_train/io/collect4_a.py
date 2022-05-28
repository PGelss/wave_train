import numpy as np
# import scipy.io as sio  # needed for mat-file output only
import pickle
import sys

class Collect4_a:
    """
    Collect and save data from four-dimensional scans
    (1) N = number of sites
    (2) R = max. ranks of solutions
    (3) I = choice of integrator
    (4) S = number of sub steps
    Performing three-dimensional analysis
    'RIS', with fixed N
    'NIS', with fixed R
    """

    def __init__(self, choice, fixed, cat_1, cat_2, cat_3):

        # Three categories, e.g. RIS, NIS, etc
        self.choice = choice
        self.fixed = fixed
        self.c1 = cat_1
        self.c2 = cat_2
        self.c3 = cat_3

        self.n1 = len(self.c1)
        self.n2 = len(self.c2)
        self.n3 = len(self.c3)

        # Open/read successful?
        self.ok = np.zeros([self.n1, self.n2, self.n3], dtype=bool)

        # CPU time, time steps
        self.cpu_total = np.zeros([self.n1, self.n2, self.n3])
        self.step_size = np.zeros([self.n1, self.n2, self.n3])
        self.sub_steps = np.zeros([self.n1, self.n2, self.n3])

        # Energy versus time
        self.nrgy_slope = np.zeros([self.n1, self.n2, self.n3])
        self.nrgy_inter = np.zeros([self.n1, self.n2, self.n3])
        self.nrgy_error = np.zeros([self.n1, self.n2, self.n3])
        self.nrgy_r_val = np.zeros([self.n1, self.n2, self.n3])
        self.nrgy_rmsd  = np.zeros([self.n1, self.n2, self.n3])

        # Norm versus time
        self.norm_slope = np.zeros([self.n1, self.n2, self.n3])
        self.norm_inter = np.zeros([self.n1, self.n2, self.n3])
        self.norm_error = np.zeros([self.n1, self.n2, self.n3])
        self.norm_r_val = np.zeros([self.n1, self.n2, self.n3])
        self.norm_rmsd  = np.zeros([self.n1, self.n2, self.n3])

        # RMSD versus time
        self.rmsd_slope = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_inter = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_error = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_r_val = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_rmsd  = np.zeros([self.n1, self.n2, self.n3])

        # Two counters for opening/reading the data files
        self.success = 0
        self.failure = 0

        # Show slopes or RMSDs
        self.show = 'rmsd'

    def __str__(self):

        if self.choice == 'RIS':
            my_string = """
------------------------------------------
Collect and save data from an N-R-I-S scan
------------------------------------------

Numbers (fixed) : {}
Ranks           : {}
Integrators     : {}
Substeps        : {}
                """.format(self.fixed, self.c1, self.c2, self.c3)
        elif self.choice == 'NIS':
            my_string = """
------------------------------------------
Collect and save data from an N-R-I-S scan
------------------------------------------
    
Numbers         : {}
Ranks (fixed)   : {}
Integrators     : {}
Substeps        : {}
                """.format(self.c1, self.fixed, self.c2, self.c3)

        else:
            sys.exit('Wrong choice! Should be RIS or NIS only.')

        return my_string

    def read(self):

        # Console ouput using __str__ method
        print(self)

        # loop over first category
        for i1 in range(self.n1):

            # loop over second category
            for i2 in range(self.n2):

                # loop over third category
                for i3 in range(self.n3):

                    if self.choice == 'RIS':
                        data_file = 'N' + self.fixed  + '_R' + self.c1[i1] + '_I' + self.c2[i2] + '_S' + self.c3[i3] + '.pic'
                    elif self.choice == 'NIS':
                        data_file = 'N' + self.c1[i1] + '_R' + self.fixed  + '_I' + self.c2[i2] + '_S' + self.c3[i3] + '.pic'

                    # Try load data from pickle file
                    try:
                        with open(data_file, 'rb') as dynamics_file:  # 'rb' stands for "read binary"
                            dynamics = pickle.load(dynamics_file)
                        print('Loading simulation data from data file : ', data_file)
                        self.success += 1

                        # Create dictionary representation of class
                        dict_repr = vars(dynamics)
                        for key, value in dict_repr.items():
                            # set attributes of this class with values loaded from pickle file
                            setattr(dynamics, key, value)

                        # Yes, open/read was successful
                        self.ok[i1, i2, i3] = True

                        # In principle, you don't have to do this every time
                        # but it could be that reading the first file doesn't work
                        # if i1 == 0 and i2 == 0 and i3 == 0:
                        # Get names of Hamiltonian and dynamics, comparison
                        self.dynamics = dynamics.name
                        self.hamilton = dynamics.hamilton.name
                        self.compare = dynamics.compare  # expecting 'pop', 'pos', or 'psi'

                        # CPU time, time steps
                        self.cpu_total[i1, i2, i3] = sum(dynamics.cput)
                        self.step_size[i1, i2, i3] = dynamics.step_size
                        self.sub_steps[i1, i2, i3] = dynamics.sub_steps

                        # Energy versus time
                        self.nrgy_slope[i1, i2, i3] = dynamics.nrgy_slope
                        self.nrgy_inter[i1, i2, i3] = dynamics.nrgy_inter
                        self.nrgy_error[i1, i2, i3] = dynamics.nrgy_error
                        self.nrgy_r_val[i1, i2, i3] = dynamics.nrgy_r_val
                        self.nrgy_rmsd [i1, i2, i3] = dynamics.nrgy_rmsd

                        # Norm versus time
                        self.norm_slope[i1, i2, i3] = dynamics.norm_slope
                        self.norm_inter[i1, i2, i3] = dynamics.norm_inter
                        self.norm_error[i1, i2, i3] = dynamics.norm_error
                        self.norm_r_val[i1, i2, i3] = dynamics.norm_r_val
                        self.norm_rmsd [i1, i2, i3] = dynamics.norm_rmsd

                        # Norm versus time
                        self.rmsd_slope[i1, i2, i3] = dynamics.rmsd_slope
                        self.rmsd_inter[i1, i2, i3] = dynamics.rmsd_inter
                        self.rmsd_error[i1, i2, i3] = dynamics.rmsd_error
                        self.rmsd_r_val[i1, i2, i3] = dynamics.rmsd_r_val
                        self.rmsd_rmsd [i1, i2, i3] = dynamics.rmsd_rmsd

                    except IOError:
                        print('******** Failed loading from data file : ', data_file)
                        self.failure += 1

                        # No, open/read was not successful
                        self.ok[i1, i2, i3] = False

        # Statistics of success vs. failure
        print(' ')
        print('Total number of pickle files expected : ', self.success + self.failure)
        print('Number of successes : ', self.success)
        print('Number of failures  : ', self.failure)
        print(' ')

    def save(self):

        # my_dict = {
        #     'choice': self.choice,
        #     'fixed': self.fixed,
        #     'cat_1': self.c1,
        #     'cat_2': self.c2,
        #     'cat_3': self.c3,
        #     'show':  self.show,
        #     'ok':    self.ok,
        #     'colors': self.colors,
        #     'dynamics': self.dynamics,
        #     'hamilton': self.hamilton,
        #     'cpu_total': self.cpu_total,
        #     'step_size': self.step_size,
        #     'sub_steps': self.sub_steps,
        #     'nrgy_slope': self.nrgy_slope,  # energy vs. time
        #     'nrgy_inter': self.nrgy_inter,
        #     'nrgy_error': self.nrgy_error,
        #     'nrgy_r_val': self.nrgy_r_val,
        #     'nrgy_rmsd': self.nrgy_rmsd,
        #     'norm_slope': self.norm_slope,  # norm vs. time
        #     'norm_inter': self.norm_inter,
        #     'norm_error': self.norm_error,
        #     'norm_r_val': self.norm_r_val,
        #     'norm_rmsd': self.norm_rmsd,
        #     'rmsd_slope': self.rmsd_slope,  # RMSD vs. time
        #     'rmsd_inter': self.rmsd_inter,
        #     'rmsd_error': self.rmsd_error,
        #     'rmsd_r_val': self.rmsd_r_val,
        #     'rmsd_rmsd': self.rmsd_rmsd,
        # }

        # Create mat-file output => Matlab, Octave
        # mat_file_name = 'show_' + self.choice + '_' + self.fixed + '.mat'
        # print(' ')
        # print('Saving collected simulation data to file : ' + mat_file_name)
        # sio.savemat(mat_file_name, my_dict, appendmat=False)

        # Create pickle-file output => Python
        pic_file_name = 'show_' + self.choice + '_' + self.fixed + '.pic'
        print('Saving collected simulation data to file : ' + pic_file_name)
        print(' ')
        pickle.dump(self, open(pic_file_name, 'wb'))
