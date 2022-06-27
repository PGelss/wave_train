import matplotlib
import numpy as np
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from wave_train.graphics.style import figure_style, apply_styles, scaling
from wave_train.graphics.helper import optimize_grid_settings, initialize_subplot_collection, movie, \
    remove_tick_duplicates, estimate_energy_limits

def basic_figure_setup(figure: Figure, style: dict):
    """
    Parameters
    ------------
    figure: Figure
        Matplotlib figure instance used for the rendering of
        dynamics results
    style: dict
        Dictionary providing general information about the appearance
        of a matplotlib figure instance. An overview of supported 
        options with opinionated default values can be found in 
        wave_train.graphics.style
    """
    if 'fig_size' in style:
        dpi = figure.get_dpi()
        width, height = style['fig_size']
        figure.set_size_inches(width / float(dpi), height / float(dpi))

    if 'fig_pos' in style:
        backend = matplotlib.get_backend()

        if backend == "TkAgg":
            figure.canvas.manager.window.wm_geometry("+%d+%d" % style['fig_pos'])
        elif backend == "WXAgg":
            figure.canvas.manager.window.SetPosition(style['fig_pos'])
        else: # Qt5 Backend
            figure.canvas.manager.window.move(*style['fig_pos'])

    if 'tight_layout' in style:
        # supported is dictionary, {} and None
        figure.tight_layout(rect=style['tight_layout'])
    else:
        figure.tight_layout(rect=figure_style["tight_layout"])

    # update tight layout every time figure is redrawn 
    figure.set_tight_layout(False)

    return figure

def configure_figure_frame(ax_frame: Axes, xlabel="", ylabel="", xcoord=(0.5, -0.07), ycoord=(-0.07, 0.5)):
    """
    Configures the outer figure frame, in which the subgrid is inserted.

    Parameters
    ------------
        ax_frame: matplotlib.Axes
            The axis belonging to the outer figure frame
        xlabel: str
            The x-label of the outer figure
        ylabel: str
            The y-label of the outer figure
        xcoord: str
            The coordinates of the xlabel. This argument is needed,
            because otherwise labels overlap with ticks from subplots in
            inner grid.
        ycoord: str
            The coordinations of the ylabel.
    """
    ax_frame.set_xticks([])
    ax_frame.set_yticks([])
    ax_frame.set_xlabel(xlabel)
    ax_frame.set_ylabel(ylabel)
    for key,spine in ax_frame.spines.items():
        spine.set_visible(False)

    ax_frame.xaxis.set_label_coords(*xcoord)
    ax_frame.yaxis.set_label_coords(*ycoord)

    return ax_frame

def logic_check_axis_limits(axes, orientation, start, end, fallback=(-0.05, 0.05)):
    """
    Logic check for axis limits, if start & end values are equivalent,
    then the fallback limits are set
    """
    if orientation == 'x':
        set_limit = axes.set_xlim
    else:
        set_limit = axes.set_ylim

    if start == end:
        start, end = fallback

    set_limit(start, end)

def adjust_axis_quantum_numbers(axes_number, n_site):
    """
    Parameters
    ------------
    axes_number: matplotlib.pyplot.Axis
        A matplotlib Axis instance to be configured for Quantum Numbers
        as displayed in plot type QuantNumbers
    n_site: uint
        An unsigned integer referencing the total number of sites in the system
    """
    # configure x-axis
    axes_number.set_xlabel("Site")
    axes_number.set_xlim(-0.25, n_site - 0.75)
    axes_number.set_xticks(np.arange(n_site))

    # configure y-axis
    axes_number.set_ylim(0, 0.6)
    axes_number.set_yticks(np.linspace(0, 0.5, 3))
    axes_number.set_ylabel("Quantum Numbers")

    return axes_number

def adjust_axis_energy(axes_energy, num_steps, e_min, e_max):
    """
    Parameters
    ------------
    axes_energy: matplotlib.pyplot.Axis
        A matplotlib Axis instance to be configured as an energy plot
    num_steps: uint
        The total number of steps/solutions 
    e_min: float
        The minimum energy 
    e_max: float
        The maximum energy

    Returns 
    ------------
    The updated axes_energy
    """
    logic_check_axis_limits(axes_energy, 'x', 0, num_steps)
    axes_energy.set_xlabel("n")
    axes_energy.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes_energy.set_ylabel("<$\psi_n$|H|$\psi_n$>")

    logic_check_axis_limits(axes_energy, 'y', e_min, e_max)
    axes_energy.set_yticks(np.linspace(e_min, e_max, 5))
    axes_energy.yaxis.tick_right()
    axes_energy.yaxis.set_label_position("right")

    return axes_energy

def adjust_axis_energy_dynamics(axes_energy, max_time, e_min, e_max, quant=True):
    logic_check_axis_limits(axes_energy, 'x', 0, max_time)

    axes_energy.set_xlabel("t")

    if quant:
        axes_energy.set_ylabel("<$\psi$(t)|H|$\psi$(t)>")
    else:
        axes_energy.set_ylabel("Energies")

    axes_energy.set_ylim(e_min, e_max)
    axes_energy.set_yticks(np.linspace(e_min, e_max, 5))
    axes_energy.yaxis.tick_right()
    axes_energy.yaxis.set_label_position("right")

    return axes_energy

def adjust_axis_norm(axes_norm, num_steps):
    """
    Parameters
    ------------
    axes_norm: matplotlib.pyplot.Axis
        A matplotlib Axis instance to be configured as a norm plot
    num_steps: uint
        The total number of steps/solutions 

    Returns 
    ------------
    The updated axes_norm
    """
    logic_check_axis_limits(axes_norm, 'x', 0, num_steps)
    axes_norm.set_xlabel("n")
    axes_norm.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes_norm.set_ylabel("<$\psi_n$|$\psi_n$>$^{1/2}$")
    axes_norm.set_ylim(0.998, 1.002)
    axes_norm.set_yticks(np.linspace(0.998, 1.002, 5))
    axes_norm.yaxis.tick_right()
    axes_norm.yaxis.set_label_position("right")

    return axes_norm

def adjust_axis_norm_dynamics(axes_norm, max_time):
    """
    Parameters
    ------------
    axes_norm: matplotlib.pyplot.Axis
        A matplotlib Axis instance to be configured as a norm plot
    num_steps: uint
        The total number of steps/solutions 

    Returns 
    ------------
    The updated axes_norm
    """
    logic_check_axis_limits(axes_norm, 'x', 0, max_time)
    axes_norm.set_ylabel("<$\psi(t)$|$\psi(t)$>$^{1/2}$")
    axes_norm.set_ylim(0.998, 1.002)
    axes_norm.set_yticks(np.linspace(0.998, 1.002, 5))
    axes_norm.yaxis.tick_right()
    axes_norm.yaxis.set_label_position("right")

    return axes_norm

def adjust_axis_autocorrelation(axes_autoc, max_time):
    logic_check_axis_limits(axes_autoc, 'x', 0, max_time)
    axes_autoc.set_xlabel("t")
    axes_autoc.set_ylabel("<$\psi$(0)|$\psi$(t)>")
    axes_autoc.set_ylim(-1.1, 1.1)
    axes_autoc.set_yticks(np.linspace(-1, 1, 5))
    axes_autoc.yaxis.tick_right()
    axes_autoc.yaxis.set_label_position("right")

    return axes_autoc

############################################################
#            CONFIGURATION FUNCTIONS TISE                  #
############################################################

def configure_expectation_values_tise(figure, dynamics, outer_grid):
    axes_energy = figure.add_subplot(outer_grid[0, 1])  # upper right
    axes_norm   = figure.add_subplot(outer_grid[1, 1])  # lower right

    axes_norm   = adjust_axis_norm(axes_norm, dynamics.num_steps)

    if hasattr(dynamics, 'exct'): # check if analytic solution exists
        axes_energy = adjust_axis_energy(axes_energy, dynamics.num_steps, dynamics.exct[0], 1.2 * dynamics.exct[-1])
    else:
        axes_energy = adjust_axis_energy(axes_energy, dynamics.num_steps, dynamics.e_min, dynamics.e_max)

    # initialize axes with default lines
    axes_norm.plot([], [], linestyle='--', marker='o', label="Numeric")
    axes_energy.plot([], [], linestyle='--', marker='o', label="Numeric")

    if hasattr(dynamics, 'exct'):
        axes_energy.plot([], [], linestyle='solid', marker='x', color='r', label="Analytic")
        axes_energy.legend()

    return axes_energy, axes_norm

def configure_quant_numbers_basic(figure, dynamics, outer_grid=None, style=figure_style):
    """
    Configuration function for plot type QuantNumbers with
    expectation values. Function will add a grid to the
    figure object of shape (1, 1).

    Returns
    ---------
    The updated figure instance
    """

    axes_number = figure.add_subplot(outer_grid[:, 0])  # left half of figure
    axes_number = adjust_axis_quantum_numbers(axes_number, dynamics.hamilton.n_site)

    has_bessel = hasattr(dynamics, 'bessel')

    axes_number.bar(np.arange(dynamics.hamilton.n_site), np.zeros(dynamics.hamilton.n_site), width=0.4)

    if has_bessel:
        axes_number.bar(0.1 + np.arange(dynamics.hamilton.n_site), np.zeros(dynamics.hamilton.n_site), 
            width=0.4, alpha=0.4, color="b")

    apply_styles([axes_number], style)
    return figure

def configure_quant_numbers2_basic(figure, dynamics, outer_grid=None, style=figure_style):

    axes_number = figure.add_subplot(outer_grid[:, 0])  # left half of figure

    axes_number.set_xlim(-0.25, dynamics.hamilton.n_site - 0.75)
    axes_number.set_xlabel("Site")

    axes_number.set_ylim(0, 2)
    axes_number.set_yticks(np.linspace(0, 2, 5))
    axes_number.set_ylabel("Quantum numbers")

    # generate initial bars of height zero for excitonic and phononic system
    bars_exc = np.arange(dynamics.hamilton.n_site) - 0.1
    bars_pho = np.arange(dynamics.hamilton.n_site) + 0.1
    heights  = np.zeros(dynamics.hamilton.n_site)

    # first n_site rectangles belong to excitonic quantum numbers only,
    # second half to phononic quantum numbers only
    axes_number.bar(bars_exc, heights, width=0.2, color='green', label="Quantum")
    axes_number.bar(bars_pho, heights, width=0.2, color='orange', label="Classical")
    axes_number.legend()

    apply_styles([axes_number], style)
    return figure

def configure_quant_displace2_basic(figure, dynamics, outer_grid=None, style=figure_style):

    axes_displace = figure.add_subplot(outer_grid[:, 0])  # left half of figure

    # x-axis settings
    axes_displace.set_xlim(-0.25, dynamics.hamilton.n_site - 0.75)
    axes_displace.set_xticks(np.arange(dynamics.hamilton.n_site))
    axes_displace.set_xlabel("Site")

    axes_displace.set_ylim(-1.0, 1.0)
    axes_displace.set_yticks(np.linspace(-1.0, 1.0, 5))
    axes_displace.set_ylabel("Quantum Numbers; Displacements")

    # generate initial bars of height zero for shifted x1 and x2
    x1 = np.arange(dynamics.hamilton.n_site) - 0.1
    x2 = np.arange(dynamics.hamilton.n_site) + 0.1
    y  = np.zeros(dynamics.hamilton.n_site)

    # create two sets of bar plots and combine them into one container
    bwidth = 0.2

    axes_displace.bar(x1, y, width=bwidth, color='green', label="Quantum")
    axes_displace.bar(x2, y, width=bwidth, color='orange', label="Classical")
    axes_displace.legend()

    apply_styles([axes_displace], style)
    return figure

def configure_positions2_basic(figure, dynamics, outer_grid=None, style=figure_style):

    axes_positions = figure.add_subplot(outer_grid[:, 0])  # left half of figure

    n_site = dynamics.hamilton.n_site
    # x-axis settings
    xticks = np.arange(0, n_site, 1)

    axes_positions.set_xlim((-1, n_site))
    axes_positions.set_xticks(xticks)

    # plot default position of the lattice sites
    site_pos = np.arange(0, n_site, 1)

    axes_positions.set_ylim(-0.1, 1)
    axes_positions.set_yticks(np.arange(0, 1.1, 0.2))

    axes_positions.set_xlabel("sites")
    axes_positions.set_ylabel("Quantum Numbers")

    axes_positions.scatter(site_pos, np.repeat(0, n_site), marker='o', facecolors='none', edgecolors='b')

    for _ in range(n_site):
        axes_positions.plot([], [], marker='x', linestyle=None, color='red')
        axes_positions.plot([], [], marker='.', linestyle='-', color='blue')

    apply_styles([axes_positions], style)
    return figure

def configure_populations_basic(figure, dynamics, outer_grid=None, style=figure_style):
    """
    Configure the plotting setup for the visualisation of populations
    for every quantum state in the basis. Figure has a grid layout,
    with the optimization of the grid being limited to a maximum of
    5 sites. The number of rows will depend on the optimal number
    of columns found for visualisation.

    Parameters
    ------------
    figure: matplotlib.figure.Figure
        The plot figure
    dynamics: instance of TISE or TDSE
        The dynamics used for visualization
    gs_opt: tuple, (default (None, None, None))
        spec: GridSpec
            The GridSpecification
        nrows: uint
            The number of rows for the basic grid setup, must 
            be configured manually when called from configure_*_expect_*
            function
        ncols: uint
            The number of cols for the basic grid setup
    Returns
    ---------
    The updated figure
    """

    ax_frame = figure.add_subplot(outer_grid[:, 0])  # left half of figure
    ax_frame = configure_figure_frame(ax_frame, r"State", r"Populations")

    # dynamically add the required number of subplots
    nrows, ncols = optimize_grid_settings(dynamics.hamilton.n_site)
    initialize_subplot_collection(figure, outer_grid, nrows, ncols, dynamics.hamilton.n_site)

    has_bessel = hasattr(dynamics, 'bessel')

    subgrid_axes = figure.axes[1:(1+dynamics.hamilton.n_site)]
    subgrid_axes = remove_tick_duplicates(subgrid_axes, nrows, ncols)

    # initialize bar plot instances
    for ax in subgrid_axes:
        ax.bar(np.arange(dynamics.hamilton.n_dim), np.zeros(dynamics.hamilton.n_dim))
        # Population values can be between 0 and 1
        ax.set_ylim(0, 1)
        # Set population values to number of dimensions
        ax.set_xticks(np.arange(0, dynamics.hamilton.n_dim))
        # Analytic Bessel function solution if available
        if has_bessel:
            ax.bar(0.1 + np.arange(dynamics.hamilton.n_dim), np.zeros(dynamics.hamilton.n_dim), alpha=0.4, color='b')

    subgrid_axes = subgrid_axes.flatten()[:dynamics.hamilton.n_site]

    apply_styles(subgrid_axes, style)
    return figure

def configure_densitymat_basic(figure, dynamics, outer_grid=None, style=figure_style):


    ax_frame = figure.add_subplot(outer_grid[:, 0])  # left half of figure
    ax_frame = configure_figure_frame(ax_frame, r"State", r"State")

    # dynamically add the required number of subplots
    nrows, ncols = optimize_grid_settings(dynamics.hamilton.n_site)
    initialize_subplot_collection(figure, outer_grid, nrows, ncols, dynamics.hamilton.n_site)

    subgrid_axes = figure.axes[1:(1+dynamics.hamilton.n_site)]
    subgrid_axes = remove_tick_duplicates(subgrid_axes, nrows, ncols)

    for ax in subgrid_axes:
        ax.imshow(np.zeros((dynamics.hamilton.n_dim, dynamics.hamilton.n_dim)))
    
    apply_styles(subgrid_axes, style)
    return figure

def configure_phasespace_basic(figure, dynamics, outer_grid=None, style=figure_style):


    ax_frame = figure.add_subplot(outer_grid[:, 0])  # left half of figure
    ax_frame = configure_figure_frame(ax_frame, r"<x>", r"<p>")

    # dynamically add the required number of subplots
    nrows, ncols = optimize_grid_settings(dynamics.hamilton.n_site)
    initialize_subplot_collection(figure, outer_grid, nrows, ncols, dynamics.hamilton.n_site)

    subgrid_axes = figure.axes[1:(1+dynamics.hamilton.n_site)]
    subgrid_axes = remove_tick_duplicates(subgrid_axes, nrows, ncols)

    for ax in subgrid_axes:
        ax.plot([], [], marker=',', linestyle='-')
        # set default axes limits; may be overwritten (plot_style)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    apply_styles(subgrid_axes, style)
    return figure

def configure_quant_numbers_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_quant_numbers_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_populations_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_populations_basic(figure, dynamics, outer_grid=outer_grid, style=style) 
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_densitymat_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_densitymat_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_phasespace_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_phasespace_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_quant_numbers2_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_quant_numbers2_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_quant_displace2_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_quant_displace2_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

def configure_positions2_expect_tise(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(2, 2, figure=figure)
    configure_positions2_basic(figure, dynamics, outer_grid=outer_grid, style=style)
    configure_expectation_values_tise(figure, dynamics, outer_grid)
    return figure

############################################################
#            CONFIGURATION FUNCTIONS TDSE                  #
############################################################

def configure_expectation_values_tdse(figure, dynamics, outer_grid):
    axes_energy = figure.add_subplot(outer_grid[0, 1])  # upper right
    axes_norm   = figure.add_subplot(outer_grid[1, 1])  # middle right
    axes_autoc  = figure.add_subplot(outer_grid[2, 1])  # lower right

    max_time = dynamics.num_steps * dynamics.step_size

    axes_norm   = adjust_axis_norm_dynamics(axes_norm, max_time)
    axes_energy = adjust_axis_energy_dynamics(axes_energy, max_time, 0, 1)
    axes_autoc  = adjust_axis_autocorrelation(axes_autoc, max_time)

    # initialize axes with default lines
    axes_norm.plot([], [], linestyle='--', marker='o', label="Numeric")
    axes_energy.plot([], [], linestyle='--', marker='o', label="Numeric")
    # autocorrelation is broken down into absolute, real and imaginary parts
    axes_autoc.plot([], [], marker='o', linestyle='--', label="Absolute")
    axes_autoc.plot([], [], marker='x', linestyle='--', label="Real")
    axes_autoc.plot([], [], marker='v', linestyle='--', label="Imaginary")

    axes_autoc.legend()

    return figure

def configure_quant_numbers_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_quant_numbers_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_populations_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_populations_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_densitymat_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_densitymat_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_phasespace_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_phasespace_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_quant_numbers2_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_quant_numbers2_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_quant_displace2_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_quant_displace2_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

def configure_positions2_expect_tdse(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)  #, height_ratios=[1, 0.6, 0.6, 0.6])
    configure_positions2_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_tdse(figure, dynamics, outer_grid)
    return figure

############################################################
#            CONFIGURATION FUNCTIONS QCMD                  #
############################################################

def configure_expectation_values_qcmd(figure, dynamics, outer_grid):
    """
    QCMD expectation values are broken down into energy, norm
    and autocorrelation. The energy subplot further yields
    information on the quantum mechanical, quantum classical
    and classical energy, which are displayed in the energy 
    plot along with the total energy.
    """
    axes_energy = figure.add_subplot(outer_grid[0, 1])  # upper right
    axes_norm   = figure.add_subplot(outer_grid[1, 1])  # middle right
    axes_autoc  = figure.add_subplot(outer_grid[2, 1])  # lower right

    max_time = dynamics.num_steps * dynamics.step_size

    # configure the axis labels
    axes_energy = adjust_axis_energy_dynamics(axes_energy, max_time, 0, 1, False)
    axes_norm   = adjust_axis_norm_dynamics(axes_norm, max_time)
    axes_autoc  = adjust_axis_autocorrelation(axes_autoc, max_time)

    axes_energy.plot([], [], linestyle='--', marker='x', label="Quantum", color='green')
    axes_energy.plot([], [], linestyle='--', marker='o', label="Coupling", color='purple')
    axes_energy.plot([], [], linestyle='--', marker='v', label="Classical", color='orange')
    axes_energy.plot([], [], linestyle='--', marker='x', label="Total")

    axes_norm.plot([], [], linestyle='--', marker='x', label="Numeric")

    axes_autoc.plot([], [], linestyle='--', marker='x', label="Real")
    axes_autoc.plot([], [], linestyle='--', marker='o', label="Imaginary")
    axes_autoc.plot([], [], linestyle='--', marker='v', label="Absolute")

    axes_energy.legend()
    axes_autoc.legend()

    return figure

def configure_quant_displace2_expect_qcmd(figure, dynamics, style=figure_style):
    """
    Average quantum numbers for each of the sites, 
    separately for the two subsystems
    ---------------------------------------------

    Special version for bipartite systems:
    Using matplotlib's bar function to create a double
    bar chart from two one-dimensional numpy arrays.
    Quantum numbers for each site are retrieved from the
    (reduced) density matrices and the number operators
    using the trace formula for expectation values.
    These operations are done separately for excitonic and phononic densities
    """
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)
    configure_quant_displace2_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_qcmd(figure, dynamics, outer_grid)
    return figure

def configure_positions2_expect_qcmd(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(3, 2, figure=figure)
    configure_positions2_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_qcmd(figure, dynamics, outer_grid)
    return figure

############################################################
#            CONFIGURATION FUNCTIONS CeoM                  #
############################################################

def configure_expectation_values_ceom(figure, dynamics, outer_grid):
    # energy is the only expectation value 
    axes_energy = figure.add_subplot(outer_grid[:, 1])

    max_time = dynamics.num_steps * dynamics.step_size

    axes_energy = adjust_axis_energy_dynamics(axes_energy, max_time, 0, 1, False)

    axes_energy.plot([], [], linestyle='--', marker='x', label="Potential")
    axes_energy.plot([], [], linestyle='--', marker='x', label="Kinetic")
    axes_energy.plot([], [], linestyle='--', marker='o', label="Total")

    axes_energy.legend()
    return figure

def configure_phasespace_expect_ceom(figure, dynamics, style=figure_style):
    outer_grid = gridspec.GridSpec(1, 2, figure=figure)
    configure_phasespace_basic(figure, dynamics, outer_grid, style=style)
    configure_expectation_values_ceom(figure, dynamics, outer_grid)
    return figure

############################################################
#                 UPDATE FUNCTIONS TISE                    #
############################################################

@movie
def update_quant_numbers_basic(i, figure, dynamics, writer, saving):
    """
    Function updates the quantum numbers of the QuantNumbers plot
    by retrieving the first axis of the plot and plotting the quantum
    number as a bar plot.

    Parameters
    ------------
    i: uint
        Current animation step
    figure: 
        Figure object holding the plot
    dynamics:
        dynamics instance used for obtaining the next solution
    writer: Animation
        The Animation instance used for saving animated output
    saving: bool
        If True, save_as_png is called for creating animated output
    """
    dynamics.update_solve(i)

    # get the axes for plotting the quantum numbers, by definition first axis
    axes_number = figure.axes[0]

    has_bessel = hasattr(dynamics, 'bessel')

    if has_bessel:
        last = len(axes_number.patches) // 2

        for j, rectangle in enumerate(axes_number.patches[last:]):
            rectangle.set_height(dynamics.bessel[i][j])
    else:
        last = len(axes_number.patches)

    for j, rectangle in enumerate(axes_number.patches[:last]):
        rectangle.set_height(dynamics.qu_numbr[i, j])


    figure.suptitle(dynamics.head[i])

@movie
def update_quant_numbers2_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    # get all rectangle instances of the first plot
    # first half of array corresponds to excitonic values
    # second half of array corresponds to phononic values
    rectangles = figure.axes[0].patches

    for k in range(dynamics.hamilton.n_site):
        rectangles[k].set_height(dynamics.ex_numbr[i, k])
        rectangles[k + dynamics.hamilton.n_site].set_height(dynamics.ph_numbr[i, k])

    figure.suptitle(dynamics.head[i])

@movie
def update_quant_displace2_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    rectangles = figure.axes[0].patches

    scale = scaling()

     # split densities and get expectation values of number operators
    for j in range(dynamics.hamilton.n_site):
        rectangles[j].set_height(dynamics.ex_numbr[i, j])
        height = scale * dynamics.position[i, j]

        rectangles[j + dynamics.hamilton.n_site].set_height(height)

    figure.suptitle(dynamics.head[i])

@movie
def update_positions2_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    pos = scaling() * dynamics.position[i]
    nex = dynamics.ex_numbr[i]

    # all even indices correspond to phononic system
    phononic    = np.asarray(figure.axes[0].lines)[0::2]
    # all odd indices correspond to excitonic system
    excitonic   = np.asarray(figure.axes[0].lines)[1::2]

    for j, (line_ph, line_ex) in enumerate(zip(phononic, excitonic)):
        line_ph.set_data([j, j + pos[j]], [0, 0])
        # move excitonic quantum number with point of displacement
        line_ex.set_data([j + pos[j], j + pos[j]], [0, nex[j]])

    figure.suptitle(dynamics.head[i])

@movie
def update_populations_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    populations = np.asarray([np.diag(x) for x in dynamics.rho_site[i]])

    has_bessel = hasattr(dynamics, 'bessel')

    # iterate over all available axes, start at index 1, because of subgrid 
    for k, ax in enumerate(figure.axes[1:dynamics.hamilton.n_site + 1]):
        for rectangle, pop in zip(ax.patches[:dynamics.hamilton.n_dim], populations[k]):
            # update the population values for every quantum state
            rectangle.set_height(pop)

        if has_bessel: # if analytic solution with Bessel function exists, then visualize
            states = np.repeat(np.array([[1, 0]], dtype=float), dynamics.hamilton.n_site, axis=0)
            bessel = dynamics.bessel[i]
            # calculate the population values for the ground and excited state
            bessel_population = np.abs(states - np.column_stack((bessel, bessel)))
            for rectangle, bessel_pop in zip(ax.patches[-dynamics.hamilton.n_dim:], bessel_population[k]):
                # update the population values for every quantum state
                rectangle.set_height(bessel_pop)
        
    figure.suptitle(dynamics.head[i])

@movie
def update_densitymat_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    densities = np.asarray([np.abs(x) for x in dynamics.rho_site[i]])

    for j, ax in enumerate(figure.axes[1:dynamics.hamilton.n_site + 1]):
        ax.clear()
        ax.imshow(densities[j])

    figure.suptitle(dynamics.head[i])

@movie
def update_phasespace_basic(i, figure, dynamics, writer, saving):
    dynamics.update_solve(i)

    for j, ax in enumerate(figure.axes[1:dynamics.hamilton.n_site + 1]):
        position = dynamics.position[:i+1, j]
        momentum = dynamics.momentum[:i+1, j]
        ax.lines[0].set_data(position, momentum)

    figure.suptitle(dynamics.head[i])

def update_expectation_values_tise(i, figure, dynamics):
    axes_energy, axes_norm = figure.axes[-2], figure.axes[-1]

    axes_energy.lines[0].set_data(dynamics.time[:i+1], dynamics.nrgy[:i+1])

    if hasattr(dynamics, 'exct'):
        axes_energy.lines[1].set_data(dynamics.time[:i+1], dynamics.exct[:i+1])

    axes_norm.lines[0].set_data(dynamics.time[:i+1], dynamics.norm[:i+1])

@movie
def update_quant_numbers_expect_tise(i, figure, dynamics, writer, saving):
    # update the quantum numbers
    update_quant_numbers_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_populations_expect_tise(i, figure, dynamics, writer, saving):
    update_populations_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_densitymat_expect_tise(i, figure, dynamics, writer, saving):
    update_densitymat_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_phasespace_expect_tise(i, figure, dynamics, writer, saving):
    update_phasespace_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_quant_numbers2_expect_tise(i, figure, dynamics, writer, saving):
    update_quant_numbers2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_quant_displace2_expect_tise(i, figure, dynamics, writer, saving):
    update_quant_displace2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

@movie
def update_positions2_expect_tise(i, figure, dynamics, writer, saving):
    update_positions2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tise(i, figure, dynamics)

############################################################
#                 UPDATE FUNCTIONS TDSE                    #
############################################################
def update_expectation_values_tdse(i, figure, dynamics):
    axes_energy, axes_norm, axes_autoc = figure.axes[-3], figure.axes[-2], figure.axes[-1]

    if i == 0:
        init = dynamics.nrgy[0]

        if init > 0:
            ticks = np.linspace(0.998 * init, 1.002 * init, 5, endpoint=True)
        else:
            ticks = np.linspace(1.002 * init, 0.998 * init, 5, endpoint=True)

        axes_energy.set_ylim(ticks[0], ticks[-1])
        axes_energy.set_yticks(ticks)

    time = dynamics.time[:i+1]
    # update energy expectation values
    axes_energy.lines[0].set_data(time, dynamics.nrgy[:i+1])
    # update norm expectation values
    axes_norm.lines[0].set_data(time, dynamics.norm[:i+1])
    # update autocorrelation expectation values
    axes_autoc.lines[0].set_data(time, np.abs(dynamics.auto[:i+1]))
    axes_autoc.lines[1].set_data(time, np.real(dynamics.auto[:i+1]))
    axes_autoc.lines[2].set_data(time, np.imag(dynamics.auto[:i+1]))

@movie
def update_quant_numbers_expect_tdse(i, figure, dynamics, writer, saving):
    update_quant_numbers_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_populations_expect_tdse(i, figure, dynamics, writer, saving):
    update_populations_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_densitymat_expect_tdse(i, figure, dynamics, writer, saving):
    update_densitymat_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_phasespace_expect_tdse(i, figure, dynamics, writer, saving):
    update_phasespace_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_quant_numbers2_expect_tdse(i, figure, dynamics, writer, saving):
    update_quant_numbers2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_quant_displace2_expect_tdse(i, figure, dynamics, writer, saving):
    update_quant_displace2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

@movie
def update_positions2_expect_tdse(i, figure, dynamics, writer, saving):
    update_positions2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_tdse(i, figure, dynamics)

############################################################
#                 UPDATE FUNCTIONS QCMD                    #
############################################################
def update_expectation_values_qcmd(i, figure, dynamics):
    axes_energy, axes_norm, axes_autoc = figure.axes[-3], figure.axes[-2], figure.axes[-1]

    if i == 0:  # rescale energy axis
        energies = np.concatenate((dynamics.e_quant[:i+1], dynamics.e_qu_cl[:i+1], dynamics.e_class[:i+1], dynamics.nrgy[:i+1]))
        energy_min, energy_max = estimate_energy_limits(energies)
        axes_energy.set_ylim(energy_min, energy_max)
        axes_energy.set_yticks(np.linspace(energy_min, energy_max, 4, endpoint=True))

    # update times
    time = dynamics.time[:i+1]

    # update energy expectation values
    axes_energy.lines[0].set_data(time, dynamics.e_quant[:i+1])
    axes_energy.lines[1].set_data(time, dynamics.e_qu_cl[:i+1])
    axes_energy.lines[2].set_data(time, dynamics.e_class[:i+1])
    axes_energy.lines[3].set_data(time, dynamics.nrgy[:i+1])

    # update norm expectation values
    axes_norm.lines[0].set_data(time, dynamics.norm[:i+1])

    # update autocorrelation expectation values
    axes_autoc.lines[0].set_data(time, np.real(dynamics.auto[:i+1]))
    axes_autoc.lines[1].set_data(time, np.imag(dynamics.auto[:i+1]))
    axes_autoc.lines[2].set_data(time, np.abs(dynamics.auto[:i+1]))

@movie
def update_quant_displace2_expect_qcmd(i, figure, dynamics, writer, saving):
    update_quant_displace2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_qcmd(i, figure, dynamics)

@movie
def update_positions2_expect_qcmd(i, figure, dynamics, writer, saving):
    update_positions2_basic(i, figure, dynamics, writer, False)
    update_expectation_values_qcmd(i, figure, dynamics)

############################################################
#                 UPDATE FUNCTIONS CeoM                    #
############################################################
def update_expectation_values_ceom(i, figure, dynamics):
    axes_energy = figure.axes[-1]

    if i == 0: # rescale energy axis
        energies = np.concatenate((dynamics.potential[:i+1], dynamics.kinetic[:i+1], dynamics.nrgy[:i+1]))
        energy_min, energy_max = estimate_energy_limits(energies)
        axes_energy.set_ylim(energy_min, energy_max)
        axes_energy.set_yticks(np.linspace(energy_min, energy_max, 4, endpoint=True))

    time = dynamics.time[:i+1]
    axes_energy.lines[0].set_data(time, dynamics.potential[:i+1])
    axes_energy.lines[1].set_data(time, dynamics.kinetic[:i+1])
    axes_energy.lines[2].set_data(time, dynamics.nrgy[:i+1])

@movie
def update_phasespace_expect_ceom(i, figure, dynamics, writer, saving):
    update_phasespace_basic(i, figure, dynamics, writer, False)
    update_expectation_values_ceom(i, figure, dynamics)