import numpy as np
import scipy.io as sio
import pickle
import sys
import matplotlib.pyplot as plt
from scipy import stats

class Collect3:

    def __init__(self, choice, cat_1, cat_2, cat_3):

        # Three categories, e.g. RIS, NIS, etc
        self.choice = choice
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

        # Norm versus time
        self.norm_slope = np.zeros([self.n1, self.n2, self.n3])
        self.norm_inter = np.zeros([self.n1, self.n2, self.n3])
        self.norm_error = np.zeros([self.n1, self.n2, self.n3])
        self.norm_r_val = np.zeros([self.n1, self.n2, self.n3])

        # RMSD versus time
        self.rmsd_slope = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_inter = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_error = np.zeros([self.n1, self.n2, self.n3])
        self.rmsd_r_val = np.zeros([self.n1, self.n2, self.n3])

        # Two counters for opening/reading the data files
        self.success = 0
        self.failure = 0

        # Get matplotlib's default color scheme
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def __str__(self):

        my_string = """
------------------------------
Collect data from a R-I-S scan
------------------------------

Ranks       : {}
Integrators : {}
Substeps    : {}
                """.format(self.c1, self.c2, self.c3)

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
                        data_file = 'R' + self.c1[i1] + '_I' + self.c2[i2] + '_S' + self.c3[i3] + '.pickle'
                    elif self.choice == 'NIS':
                        data_file = 'N' + self.c1[i1] + '_I' + self.c2[i2] + '_S' + self.c3[i3] + '.pickle'
                    else:
                        sys.exit('Wrong choice: should be NIS or RIS')

                    # Try load data from pickle file
                    try:
                        with open(data_file, 'rb') as equation_file:  # 'rb' stands for "read binary"
                            equation = pickle.load(equation_file)
                        print('Loading simulation data from data file : ', data_file)
                        self.success += 1

                        # Create dictionary representation of class
                        dict_repr = vars(equation)
                        for key, value in dict_repr.items():
                            # set attributes of this class with values loaded from pickle file
                            setattr(equation, key, value)

                        # Yes, open/read was successful
                        self.ok[i1, i2, i3] = True

                        # CPU time, time steps
                        self.cpu_total[i1, i2, i3] = sum(equation.cput)
                        self.step_size[i1, i2, i3] = equation.step_size
                        self.sub_steps[i1, i2, i3] = equation.sub_steps

                        # Energy versus time
                        self.nrgy_slope[i1, i2, i3] = equation.nrgy_slope
                        self.nrgy_inter[i1, i2, i3] = equation.nrgy_inter
                        self.nrgy_error[i1, i2, i3] = equation.nrgy_error
                        self.nrgy_r_val[i1, i2, i3] = equation.nrgy_r_val

                        # Norm versus time
                        self.norm_slope[i1, i2, i3] = equation.norm_slope
                        self.norm_inter[i1, i2, i3] = equation.norm_inter
                        self.norm_error[i1, i2, i3] = equation.norm_error
                        self.norm_r_val[i1, i2, i3] = equation.norm_r_val

                        # Norm versus time
                        self.rmsd_slope[i1, i2, i3] = equation.rmsd_slope
                        self.rmsd_inter[i1, i2, i3] = equation.rmsd_inter
                        self.rmsd_error[i1, i2, i3] = equation.rmsd_error
                        self.rmsd_r_val[i1, i2, i3] = equation.rmsd_r_val

                    except IOError:
                        print('******** Failed loading from data file : ', data_file)
                        self.failure += 1

                        # No, open/read was not successful
                        self.ok[i1, i2, i3] = False

    def save(self):

        my_dict = {
            'cat_1': self.c1,
            'cat_2': self.c2,
            'cat_3': self.c3,
            'cpu_total': self.cpu_total,
            'step_size': self.step_size,
            'sub_steps': self.sub_steps,
            'nrgy_slope': self.nrgy_slope,
            'nrgy_inter': self.nrgy_inter,
            'nrgy_error': self.nrgy_error,
            'nrgy_r_val': self.nrgy_r_val,
            'norm_slope': self.norm_slope,
            'norm_inter': self.norm_inter,
            'norm_error': self.norm_error,
            'norm_r_val': self.norm_r_val,
            'rmsd_slope': self.rmsd_slope,
            'rmsd_inter': self.rmsd_inter,
            'rmsd_error': self.rmsd_error,
            'rmsd_r_val': self.rmsd_r_val
        }

        # Create mat-file output => Matlab, Octave
        mat_file_name = 'collect_' + self.choice + '.mat'
        print(' ')
        print('Saving collected simulation data to file : ' + mat_file_name)
        sio.savemat(mat_file_name, my_dict, appendmat=False)

        # Create pickle-file output => Python
        picke_file_name = 'collect_' + self.choice + '.pickle'
        print('Saving collected simulation data to file : ' + picke_file_name)
        print(' ')
        pickle.dump(my_dict, open(picke_file_name, 'wb'))

        # Statistics of success vs. failure
        print('Total number of pickle files expected : ', self.success + self.failure)
        print('Number of successes : ', self.success)
        print('Number of failures  : ', self.failure)

    def plot(self):

        # Create a new figure and set its size
        # TODO in which units?
        plt.figure(figsize=(7, 10))

        # Common x-axes for all subfigures
        x_min = np.min(self.step_size[self.ok] / self.sub_steps[self.ok])
        x_max = np.max(self.step_size[self.ok] / self.sub_steps[self.ok])
        x_min = 10**np.floor(np.log10(x_min))
        x_max = 10**np.ceil(np.log10(x_max))

        print(' ')
        print('*************************')
        print('CPU time versus step size')
        print('*************************')
        y_min = np.min(self.cpu_total[self.ok]) / 2
        y_max = np.max(self.cpu_total[self.ok]) * 2
        print (' ')
        print ('x-range : from ', x_min, ' to ', x_max)
        print ('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            print('ranks = ' + self.c1[i1])

            plt.subplot(4, self.n1+1, i1+1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                y = self.cpu_total[i1, i2, m]
                self.my_loglog(x, y, i2)

            # One day, one hour, one minute
            x1 = np.array([x_min, x_max])
            d1 = np.ones(2) * 60 ** 2 * 24
            h1 = np.ones(2) * 60 ** 2
            m1 = np.ones(2) * 60
            s1 = np.ones(2)
            if self.cpu_total.any() > d1[0]:
                plt.loglog(x1, d1, 'k:', label="1 day")
            plt.loglog(x1, h1, 'k-', label="1 hour")
            if self.cpu_total.any() < 120:
                plt.loglog(x1, m1, 'k-.', label="1 min.")
            if self.cpu_total.any() < 2:
                plt.loglog(x1, s1, 'k--', label="1 sec.")

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                plt.ylabel('CPU time [sec.]')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

            # Figure title and legend
            if i1 == self.n1 - 1:  # last column only
                # plt.legend(loc='best')
                plt.legend(title='Integrators', bbox_to_anchor=(1.5, 1))
            plt.title('ranks = ' + self.c1[i1])

        print(' ')
        print('*****************************************')
        print('Rel. drift of energy versus step size')
        print('*****************************************')
        y = self.nrgy_slope[self.ok] / self.nrgy_inter[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2
        print (' ')
        print ('x-range : from ', x_min, ' to ', x_max)
        print ('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            print('ranks = ' + self.c1[i1])

            plt.subplot(4, self.n1+1, self.n1+1+i1+1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                y = self.nrgy_slope[i1, i2, m] / self.nrgy_inter[i1, i2, m]
                self.my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                plt.ylabel('|relative drift of energy|')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        print(' ')
        print('*****************************************')
        print('Rel. drift of norm versus step size')
        print('*****************************************')
        y = self.norm_slope[self.ok] / self.norm_inter[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2
        print (' ')
        print ('x-range : from ', x_min, ' to ', x_max)
        print ('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            print('ranks = ' + self.c1[i1])

            plt.subplot(4, self.n1+1, 2*(self.n1+1)+i1+1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                y = self.norm_slope[i1, i2, m] / self.norm_inter[i1, i2, m]
                self.my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                plt.ylabel('|relative drift of norm|')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        print(' ')
        print('*****************************************')
        print('Drift of RMSD versus step size')
        print('*****************************************')
        y = self.rmsd_slope[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2

        print (' ')
        print ('x-range : from ', x_min, ' to ', x_max)
        print ('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            print('ranks = ' + self.c1[i1])

            plt.subplot(4, self.n1+1, 3*(self.n1+1)+i1+1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                y = self.rmsd_slope[i1, i2, m]
                self.my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                plt.ylabel('|absolute drift of rmsd|')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])

            plt.xlabel('time step')
            self.my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        plt.suptitle('N = 5')
        plt.show()

    def my_xticks(self, x_min, x_max):
        plt.xticks([1, 10, 100, 1000])
        l_min = np.floor(np.log10(x_min))
        l_max = np.ceil(np.log10(x_max))
        ticks = np.arange(start=l_min+1, stop=l_max)
        plt.xticks(10**ticks)

    def my_loglog(self, x, y, i2):

        # Eliminate NaN entries
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # Plot original data (double-logarithmic representation)
        plt.loglog(x[y > 0], +y[y > 0], 'o', color=self.colors[i2], label=self.c2[i2].upper())
        plt.loglog(x[y < 0], -y[y < 0], 'v', color=self.colors[i2])

        # Fit a power law
        lx = np.log10(x)
        ly = np.log10(np.abs(y))
        slope, inter, r_val, p_val, error = stats.linregress(lx,ly)
        plt.loglog(x, 10**inter * x**slope, '-', color=self.colors[i2])
        # TODO : only include data into the fit that exceed a certain "epsilon", e.g. np.finfo(float).eps

        # Console output
        m = len(x)  # total number of values
        n = sum(y < 0)  # number of negative values
        print('integrator ' + self.c2[i2].upper() + ' : ' + str(m) + ' : ' + str(n)+ ' : ' + str(slope))

