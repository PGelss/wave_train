from wave_train.hamilton.phonon import Phonon
from wave_train.dynamics.tdse import TDSE
from wave_train.io.logging import TeeLogger
from os.path import basename, splitext

def phonon_tdse(batch_mode):
    # Detect name of this script file (without extension)
    base_name   = basename(__file__)
    my_file     = splitext(base_name)[0]

    # logging instance: will be initialized with 
    # class for logging to both console and logfile
    logger = None
    if not batch_mode:
        logger = TeeLogger(log_file=my_file + ".log")

    # Set up the phononic Hamiltonian for a chain
    hamilton = Phonon(
        n_site=15,                       # number of sites
        periodic=False,                  # periodic boundary conditions
        homogen=True,                    # homogeneous chain/ring
        mass=1,                          # particle masses
        nu=1e-3,                         # position restraints
        omg=1e-3 * 2**(1/2)              # nearest neighbours
    )

    # Set up TT representation of the Hamiltonian
    hamilton.get_TT(
        n_basis=8,                       # size of HO basis set
        qtt=False,                       # using quantized TT format
    )

    # Set up TDSE solver
    dynamics = TDSE(
        hamilton=hamilton,               # choice of Hamiltonian, see above
        num_steps=50,                    # number of main time steps
        step_size=200,                   # size of main time steps
        sub_steps=50,                    # number of sub steps
        solver='s2',                     # can be 's2' (symmetrized Euler) or 'sm' (Strang-Marchuk splitting) or ...
        normalize=0,                     # whether|how to normalize the solution, can be 0|2
        max_rank=15,                     # max rank of solution
        repeats=15,                      # number of sweeps (implicit ODE solvers only!)
        threshold=1e-12,                 # threshold in ALS decomposition
        save_file=my_file+'.pic',        # if not None, generated data will be saved to this file
        load_file=None,                  # if not None, reference data will be loaded from this file
        compare=None                     # How to do the comparison with reference data
    )

    # Set up an initial state: coherent state
    dynamics.coherent(
        displace=[20.0 if i == hamilton.n_site//2 else 0.0 for i in range(hamilton.n_site)]
    )

    # Batch mode
    if batch_mode:
        dynamics.solve()                 # Solve TDSE *without* visualization

    # Interactive mode: Setup animated visualization
    else:
        from wave_train.graphics.factory import VisualTDSE
        graphics = VisualTDSE(
            dynamics=dynamics,           # choice of dynamics (EoM), see above
            plot_type='PhaseSpace',      # select your favorite plot type
            plot_expect=True,            # toggle plotting of expectation values
            figure_pos=(100, 50),        # specifying position (x,y) of upper left of figure [in pixels]
            figure_size=(1050, 450),     # specifying size (w,h) of figure [in pixels]
            image_file=my_file+'.png',   # if not None, image (last frame) will be written to this file
            movie_file=my_file+'.mp4',   # if not None, animation will be written to this file
            snapshots=False,             # save each snapshot
            frame_rate=1,                # Frames per second in mp4 animation file
            plot_style={'xlimits': (-25.0, 25.0), 'ylimits': (-0.05, 0.05)}   # for PhaseSpace only
        ).create()
        graphics.solve()                 # Solve TDSE *with* visualization


if __name__ == '__main__':
    phonon_tdse(batch_mode=False)
