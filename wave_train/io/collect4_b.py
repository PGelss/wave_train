import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from scipy import stats

class Collect4_b:
    """
    Load and visualize data from four-dimensional scans
    (1) N = number of sites
    (2) R = max. ranks of solutions
    (3) I = choice of integrator
    (4) S = number of sub steps
    Performing three-dimensional analysis
    'RIS', with fixed N
    'NIS', with fixed R
    """

    def __init__(self, choice, fixed):

        self.choice = choice
        self.fixed = fixed

        # Load pickle-file
        pic_file_name = 'show_' + self.choice + '_' + self.fixed + '.pic'
        print('Loading collected simulation data from file : ' + pic_file_name)
        print(' ')
        data = pickle.load(open(pic_file_name, 'rb'))  # 'rb' stands for "read binary"

        # Create dictionary representation of class
        dict_repr = vars(data)
        for key, value in dict_repr.items():
            # set attributes of this class with values loaded from pickle file
            setattr(self, key, value)

        # Statistics of success vs. failure
        print('Total number of pickle files expected : ', self.success + self.failure)
        print('Number of successes : ', self.success)
        print('Number of failures  : ', self.failure)

    def __str__(self):

        if self.choice == 'RIS':
            string = """
-------------------------------------------
Load and visualize data from a N-R-I-S scan
-------------------------------------------

Numbers (fixed) : {}
Ranks           : {}
Integrators     : {}
Substeps        : {}
                """.format(self.fixed, self.c1, self.c2, self.c3)
        elif self.choice == 'NIS':
            string = """
-------------------------------------------
Load and visualize data from a N-R-I-S scan
-------------------------------------------

Numbers         : {}
Ranks (fixed)   : {}
Integrators     : {}
Substeps        : {}
                """.format(self.c1, self.fixed, self.c2, self.c3)

        else:
            sys.exit('Wrong choice! Should be RIS or NIS only.')

        return string

    def plot_1(self):

        # Generate console output
        print(self)

        # Get matplotlib's default color scheme
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Create a new figure with title and set its size
        if self.choice == 'RIS':
            title = self.hamilton + ' (' + self.dynamics + '): N = ' + self.fixed
        elif self.choice == 'NIS':
            title = self.hamilton + ' (' + self.dynamics + '): r = ' + self.fixed
        plt.figure(title, figsize=(10, 10))  # Width, height in inches.

        # Common x-axes for all subfigures: Size of sub-steps
        x_min = np.min(self.step_size[self.ok] / self.sub_steps[self.ok])
        x_max = np.max(self.step_size[self.ok] / self.sub_steps[self.ok])
        x_min = 10 ** np.floor(np.log10(x_min))
        x_max = 10 ** np.ceil(np.log10(x_max))

        print(' ')
        print('*************************')
        print('CPU time versus step size')
        print('*************************')
        y_min = np.min(self.cpu_total[self.ok]) / 2
        y_max = np.max(self.cpu_total[self.ok]) * 2
        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)
        print('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            if self.choice == 'RIS':
                print('ranks = ' + self.c1[i1])
            elif self.choice == 'NIS':
                print('N = ' + self.c1[i1])

            plt.subplot(4, self.n1 + 1, i1 + 1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                y = self.cpu_total[i1, i2, m]
                self.__my_loglog(x, y, i2)

            # One day, one hour, one minute
            x1 = np.array([x_min, x_max])
            d1 = np.ones(2) * 60 ** 2 * 24
            h1 = np.ones(2) * 60 ** 2
            m1 = np.ones(2) * 60
            s1 = np.ones(2)
            if y_max > d1[0]:
                plt.loglog(x1, d1, 'k:', label="1 day")
            plt.loglog(x1, h1, 'k-', label="1 hour")
            if y_min < 120:
                plt.loglog(x1, m1, 'k-.', label="1 min.")
            if y_min < 2:
                plt.loglog(x1, s1, 'k--', label="1 sec.")

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                plt.ylabel('CPU time [sec.]')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.__my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

            # Figure title and legend
            if i1 == self.n1 - 1:  # last column only
                # plt.legend(loc='best')
                plt.legend(title='Integrators', bbox_to_anchor=(1.5, 1))
            if self.choice == 'RIS':
                plt.title('r = ' + self.c1[i1])
            elif self.choice == 'NIS':
                plt.title('N = ' + self.c1[i1])

        print(' ')
        print('*********************')
        print('Norm versus step size')
        print('*********************')
        if self.show == 'slope':
            y = self.norm_slope[self.ok] / self.norm_inter[self.ok]
        elif self.show == 'rmsd':
            y = self.norm_rmsd[self.ok] / self.norm_inter[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2
        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)
        print('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            if self.choice == 'RIS':
                print('ranks = ' + self.c1[i1])
            elif self.choice == 'NIS':
                print('N = ' + self.c1[i1])

            plt.subplot(4, self.n1 + 1, self.n1 + 1 + i1 + 1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                if self.show == 'slope':
                    y = self.norm_slope[i1, i2, m] / self.norm_inter[i1, i2, m]
                elif self.show == 'rmsd':
                    y = self.norm_rmsd[i1, i2, m] / self.norm_inter[i1, i2, m]
                self.__my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                if self.show == 'slope':
                    plt.ylabel('drift of norm')
                elif self.show == 'rmsd':
                    plt.ylabel('RMSD of norm')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.__my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        print(' ')
        print('***********************')
        print('Energy versus step size')
        print('***********************')
        if self.show == 'slope':
            y = self.nrgy_slope[self.ok] / self.nrgy_inter[self.ok]
        elif self.show == 'rmsd':
            y = self.nrgy_rmsd[self.ok] / self.nrgy_inter[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2
        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)
        print('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            if self.choice == 'RIS':
                print('ranks = ' + self.c1[i1])
            elif self.choice == 'NIS':
                print('N = ' + self.c1[i1])

            plt.subplot(4, self.n1 + 1, 2 * (self.n1 + 1) + i1 + 1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                if self.show == 'slope':
                    y = self.nrgy_slope[i1, i2, m] / self.nrgy_inter[i1, i2, m]
                elif self.show == 'rmsd':
                    y = self.nrgy_rmsd[i1, i2, m] / self.nrgy_inter[i1, i2, m]
                self.__my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                if self.show == 'slope':
                    plt.ylabel('drift of energy')
                elif self.show == 'rmsd':
                    plt.ylabel('RMSD of energy')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])
            self.__my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        print(' ')
        print('*********************')
        print('RMSD versus step size')
        print('*********************')
        if self.show == 'slope':
            y = self.rmsd_slope[self.ok]
        elif self.show == 'rmsd':
            y = self.rmsd_rmsd[self.ok]
        y_min = np.min(np.abs(y[~np.isnan(y)])) / 2
        y_max = np.max(np.abs(y[~np.isnan(y)])) * 2

        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)
        print('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            if self.choice == 'RIS':
                print('ranks = ' + self.c1[i1])
            elif self.choice == 'NIS':
                print('N = ' + self.c1[i1])

            plt.subplot(4, self.n1 + 1, 3 * (self.n1 + 1) + i1 + 1)

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.step_size[i1, i2, m] / self.sub_steps[i1, i2, m]
                if self.show == 'slope':
                    y = self.rmsd_slope[i1, i2, m]
                elif self.show == 'rmsd':
                    y = self.rmsd_rmsd[i1, i2, m]
                self.__my_loglog(x, y, i2)

            # Figure axes, labels, ticks
            if self.compare == 'pop':
                compare = 'populations'
            elif self.compare == 'pos':
                compare = 'positions'
            elif self.compare == 'psi':
                compare = 'solutions'
            else:
                sys.exit('Wrong choice of comparisons')

            if i1 == 0:  # first column only
                if self.show == 'slope':
                    plt.ylabel('drift of '+compare)
                elif self.show == 'rmsd':
                    plt.ylabel('RMSD of '+compare)
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])

            plt.xlabel('time step')
            self.__my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

        # Adding a "super title"
        # plt.suptitle(title)

        # Save graphics as a PDF file
        pdf_file_name = 'show_' + self.choice + '_' + self.fixed + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file_name)
        plt.savefig(pdf_file_name)
        # TODO: PDF options:
        #  orientation='portrait',
        #  papertype=None,
        #  format=None,

        # Show graphics on screen
        plt.show()

    def __my_xticks(self, x_min, x_max):
        plt.xticks([1, 10, 100, 1000])
        l_min = np.floor(np.log10(x_min))
        l_max = np.ceil(np.log10(x_max))
        ticks = np.arange(start=l_min + 1, stop=l_max)
        plt.xticks(10 ** ticks)

    def __my_loglog(self, x, y, i2):

        # Eliminate NaN entries
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        # Plot original data (double-logarithmic representation)
        plt.loglog(x[y > 0], +y[y > 0], 'o', color=self.colors[i2], label=self.c2[i2].upper())
        plt.loglog(x[y < 0], -y[y < 0], 'v', color=self.colors[i2])

        # Fit a power law
        lx = np.log10(x)
        ly = np.log10(np.abs(y))
        slope, inter, r_val, p_val, error = stats.linregress(lx, ly)
        # plt.loglog(x, 10 ** inter * x ** slope, '-', color=self.colors[i2])
        # TODO : only include data into the fit that exceed a certain "epsilon", e.g. np.finfo(float).eps

        # Console output
        m = len(x)  # total number of values
        n = sum(y < 0)  # number of negative values
        print('integrator ' + self.c2[i2].upper() + ' : ' + str(m) + ' : ' + str(n) + ' : ' + str(slope))

    def plot_2(self):

        # Generate console output
        print(self)

        # Get matplotlib's default color scheme
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Create a new figure with title and set its size
        if self.choice == 'RIS':
            title = self.hamilton + ' (' + self.dynamics + '): N = ' + self.fixed
        elif self.choice == 'NIS':
            title = self.hamilton + ' (' + self.dynamics + '): r = ' + self.fixed
        plt.figure(title, figsize=(12, 5))  # Width, height in inches.

        print(' ')
        print('****************')
        print('x-axis: CPU time')
        print('****************')
        x_all = self.cpu_total[self.ok]
        x_min = np.min(x_all) / 2
        x_max = np.max(x_all) * 2
        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)

        print(' ')
        print('***************************')
        print('y-axis: RMSD from reference')
        print('***************************')
        if self.show == 'slope':
            y_all = self.rmsd_slope[self.ok]
        elif self.show == 'rmsd':
            y_all = self.rmsd_rmsd[self.ok]
        y_min = np.min(np.abs(y_all[~np.isnan(y_all)])) / 2
        y_max = np.max(np.abs(y_all[~np.isnan(y_all)])) * 2
        print(' ')
        print('y-range : from ', y_min, ' to ', y_max)

        for i1 in range(self.n1):
            print(' ')
            if self.choice == 'RIS':
                print('ranks = ' + self.c1[i1])
            elif self.choice == 'NIS':
                print('N = ' + self.c1[i1])

            plt.subplot(1, self.n1 + 1, i1 + 1 )

            # Double-logarithmic representation of data points
            for i2 in range(self.n2):
                m = self.ok[i1, i2, :]  # boolean mask
                x = self.cpu_total[i1, i2, m]
                if self.show == 'slope':
                    y = self.rmsd_slope[i1, i2, m]
                elif self.show == 'rmsd':
                    y = self.rmsd_rmsd[i1, i2, m]
                plt.loglog(x, y, 'o', color=self.colors[i2], label=self.c2[i2].upper())

            # Figure axes, labels, ticks
            if i1 == 0:  # first column only
                if self.show == 'slope':
                    plt.ylabel('|absolute drift of rmsd|')
                elif self.show == 'rmsd':
                    plt.ylabel('RMSD of rmsd')
            elif i1 == self.n1 - 1:
                plt.tick_params(left=False, right=True)
                plt.tick_params(labelleft=False, labelright=True)
            else:
                plt.yticks([])

            plt.xlabel('CPU time [sec.]')
            self.__my_xticks(x_min, x_max)
            plt.axis([x_min, x_max, y_min, y_max])

            # Figure title and legend
            plt.title(('r = ' + self.c1[i1]))
            if i1 == self.n1 - 1:  # last column only
                # plt.legend(loc='best')
                plt.legend(title='Integrators', bbox_to_anchor=(2, 1))

        # Adding a "super title"
        plt.suptitle(title)

        # Save graphics as a PDF file
        pdf_file_name = 'show2_' + self.choice + '_' + self.fixed + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file_name)
        plt.savefig(pdf_file_name)
        # TODO: PDF options:
        #  orientation='portrait',
        #  papertype=None,
        #  format=None,

        # Show graphics on screen
        plt.show()

    def plot_3(self):

        # Similar to plot_2 but highest rank only

        # Generate console output
        print(self)

        # Get matplotlib's default color scheme
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Open figure and set its size
        plt.figure('plot_3', figsize=(2.5, 2.5))  # Width, height in inches.
        plt.axes([0.30,0.20,0.65,0.75])           # left, bottom, width, height in normalized (0, 1) units

        print(' ')
        print('****************')
        print('x-axis: CPU time')
        print('****************')
        x_all = self.cpu_total[self.ok]
        x_min = np.min(x_all) / 2
        x_max = np.max(x_all) * 2
        print(' ')
        print('x-range : from ', x_min, ' to ', x_max)

        print(' ')
        print('***************************')
        print('y-axis: RMSD from reference')
        print('***************************')
        if self.show == 'slope':
            y_all = self.rmsd_slope[self.ok]
        elif self.show == 'rmsd':
            y_all = self.rmsd_rmsd[self.ok]
        y_min = np.min(np.abs(y_all[~np.isnan(y_all)])) / 2
        y_max = np.max(np.abs(y_all[~np.isnan(y_all)])) * 2
        print(' ')
        print('y-range : from ', y_min, ' to ', y_max)

        i1 = self.n1-1
        print(' ')
        if self.choice == 'RIS':
            print('ranks = ' + self.c1[i1])
        elif self.choice == 'NIS':
            print('N = ' + self.c1[i1])

        # Double-logarithmic representation of data points
        for i2 in range(self.n2):
            m = self.ok[i1, i2, :]  # boolean mask
            x = self.cpu_total[i1, i2, m]
            if self.show == 'slope':
                y = self.rmsd_slope[i1, i2, m]
            elif self.show == 'rmsd':
                y = self.rmsd_rmsd[i1, i2, m]
            plt.loglog(x, y, 'o', color=self.colors[i2], label=self.c2[i2].upper())

        # Figure axes, labels, ticks
        if self.show == 'slope':
            plt.ylabel('absolute drift of solutions')
        elif self.show == 'rmsd':
            plt.ylabel('RMSD of solutions')
        # elif i1 == self.n1 - 1:
        #     plt.tick_params(left=False, right=True)
        #     plt.tick_params(labelleft=False, labelright=True)
        # else:
        #     plt.yticks([])

        plt.xlabel('CPU time [sec.]')
        self.__my_xticks(x_min, x_max)
        plt.axis([x_min, x_max, y_min, y_max])

        # Save graphics as a PDF file
        pdf_file_name = 'show3_' + self.choice + '_' + self.fixed + '.pdf'
        print(' ')
        print('Saving graphics to file : ' + pdf_file_name)
        plt.savefig(pdf_file_name, bbox_inches='tight')

        # Show graphics on screen
        plt.show()



