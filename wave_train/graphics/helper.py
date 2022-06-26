import sympy
import itertools
import numpy as np
from typing import List
from matplotlib import gridspec

def movie(func):
    """
    Decorator function for saving a snapshot of the figure object
    rendering the new output obtained for solving for a new state/
    step of the dynamics instance.

    Parameters
    ------------
        func: The decorated function. Function must receive 5 positional
        arguments, with the third and fourth argument being the Animation
        instance and if output should be saved. For further information
        see wave_train.graphics.services all update functions.
    """
    def wrapper(*args, **kwargs):
        # retrieve function arguments for update function call
        i, figure, dynamics, writer, saving = args
        # execute update function 
        func(*args, **kwargs)

        if saving: # if saving is true, the save current figure snapshot
            writer.save_as_image()
        return
    return wrapper


def optimize_grid_settings(n_sites, max_cols=5):
    """
    Finds the optimum number of cols and rows to create a
    grid under the constraint that the maximum number of 
    cols must be equal to max_cols.

    Paramaters
    ------------
        n_sites: unsigned int
            The number of plots 
    """
    # non-optimized number of rows, columns
    optimize = n_sites
    max_rows = int(np.ceil(optimize / 2))

    # optimize number of rows, columns
    if sympy.isprime(n_sites):
        optimize = n_sites + 1

    rows = range(1, max_rows + 1)
    cols = range(1, max_cols + 1)

    nrows = 0
    ncols = 0

    for (i, j) in itertools.product(rows, cols):
        if i*j == optimize:
            nrows, ncols = i, j
            break
    
    return nrows, ncols


def initialize_subplot_collection(figure, outer_grid, nrows, ncols, n_plot):
    """
    Dynamically adds subplots to a figure object based on a gridspec
    instance, that was added to the figure before this function was
    called. 

    Parameters
    ------------
    figure: matplotlib.figure.Figure
        The plot used for visualization
    outer_grid: matplotlib.GridSpec
        The grid used for adding the subplots
    nrows: uint
        The number of rows in the grid
    ncols: uint
        The number of cols in the grid
    n_plot: uint
        Desired number of plots to be added

    Returns
    ---------
    The updated figure with the added subplots 
    """
    # hspace is closely related to figure size
    inner_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[:,0], hspace=0.5)

    if not sympy.isprime(n_plot):
        for (row, col) in itertools.product(range(nrows), range(ncols)):
            figure.add_subplot(inner_grid[row, col])
    else:
        for (row, col) in itertools.product(range(nrows), range(ncols)):
            if row == nrows - 1 and col == ncols - 2:
                figure.add_subplot(inner_grid[row, col])
                break
            else:
                figure.add_subplot(inner_grid[row, col])

    return figure


def remove_tick_duplicates(axes, nrows, ncols):
    """
    Function will remove ticks crowding by removing the
    xticks and yticks from plots.

    1) If plot is in first column and last row -> Show x and y-ticks
    2) If plot is in first column but not last row -> Show y-ticks only
    3) If plot is in last row but not first column -> Show x-ticks only

    Parameters
    ------------
        axes: List
            A list of AxesSubplot's corresponding to the grid of nrows & ncols
        nrows: int
            Number of rows in the grid
        ncols: int
            Number of cols in the grid

    Returns
    ---------
        The modified list of AxesSubplots in original shape
    """
    subgrid_axes = np.pad(axes, (0, nrows * ncols - len(axes)), mode="constant", constant_values=0.0)

    for i, ax_row in enumerate(subgrid_axes.reshape((nrows, ncols))):
        for j, ax in enumerate(ax_row):
            if ax == 0.0:
                continue
            else:
                if (i == nrows - 1) and (j == 0): # if in first column and last row
                    continue
                elif j == 0: # if in first column, but not in last row
                    # disable xticks
                    ax.tick_params(labelbottom=False)
                elif i == nrows - 1: # if in last row, but not in first column
                    # disable yticks
                    ax.tick_params(labelleft=False)
                else:
                    ax.tick_params(labelbottom=False, labelleft=False)

    return subgrid_axes[subgrid_axes != 0.0]


def estimate_energy_limits(values: np.array, margin=0.1):
    """
    Given a collection of values, this function calculates lower and upper values
    by finding the min and max values in the collection and setting a safety margin

        x_lower = x_min - margin * (x_max-x_min)
        x_upper = x_max + margin * (x_max-x_min)

    Parameters
    ------------
        values: np.array
            Values from which to extract the min and max values
        margin: float
            Relative size of safety margin, see_above
    """
    values = np.asarray(values).flatten()

    # get min and max values
    x_min, x_max = values.min(), values.max()

    # calculate the lower limit
    x_lower = x_min - margin * (x_max-x_min)

    # calculate the upper limit
    x_upper = x_max + margin * (x_max-x_min)

    return x_lower, x_upper