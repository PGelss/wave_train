from wave_train.hamilton.exciton import Exciton
from wave_train.dynamics.tdse import TDSE
from wave_train.io.logging import TeeLogger
from os.path import basename, splitext

def exciton_tdse(batch_mode):
    # Detect name of this script file (without extension)
    base_name   = basename(__file__)
    my_file     = splitext(base_name)[0]

    # logging instance: will be initialized with 
    # class for logging to both console and logfile
    logger = None
    if not batch_mode:
        logger = TeeLogger(log_file=my_file + ".log")

    # Set up the excitonic Hamiltonian for a chain
    hamilton = Exciton(
        n_site=15,                       # number of sites
        periodic=False,                  # periodic boundary conditions
        homogen=True,                    # Homogeneous chain/ring
        alpha=1e-1,                      # excitonic site energy
        beta=-1e-2,                      # coupling strength (NN)
        eta=0                            # constant energy offset
    )

    # Set up TT representation of the Hamiltonian
    hamilton.get_TT(
        n_basis=2,                       # size of electronic basis set
        qtt=False                        # using quantized TT format
    )

    # Set up TDSE solver
    dynamics = TDSE(
        hamilton=hamilton,               # choice of Hamiltonian, see above
        num_steps=50,                    # number of main time steps
        step_size=20,                    # size of main time steps
        sub_steps=5,                     # number of sub steps
        solver='sm',                     # can be 's2' (symmetrized Euler) or 'sm' (Strang-Marchuk splitting) or ...
        normalize=0,                     # whether|how to normalize the solution, can be 0|2
        max_rank=8,                      # max rank of solution
        repeats=15,                      # number of sweeps (implicit ODE solvers only!)
        threshold=1e-12,                 # threshold in ALS decomposition
        save_file=my_file+'.pic',        # if not None, generated data will be saved to this file
        load_file=None,                  # if not None, reference data will be loaded from this file
        compare=None                     # How to do the comparison with reference data
    )

    # Set up initial state
    dynamics.fundamental()               # fundamental excitation near center of chain
    # dynamics.gaussian()
    # dynamics.sec_hyp(w_0=0.2)

    # Batch mode
    if batch_mode:
        dynamics.solve()                 # Solve TDSE *without* visualization

    # Interactive mode: Setup animated visualization
    else:
        from wave_train.graphics.factory import VisualTDSE
        graphics = VisualTDSE(
            dynamics=dynamics,           # choice of dynamics (EoM), see above
            plot_type='Populations',     # select your favorite plot type
            plot_expect=True,            # toggle plotting of expectation values
            figure_pos=(100, 50),        # specifying position (x,y) of upper left of figure [in pixels]
            figure_size=(1050, 450),     # specifying size (w,h) of figure [in pixels]
            image_file=my_file+'.png',   # if not None, image (last frame) will be written to this file
            movie_file=my_file+'.mp4',   # if not None, animation will be written to this file
            snapshots=False,             # save each snapshot 
            frame_rate=1,                # frames per second in mp4 animation file
            plot_style={},               # additional plot style information
        ).create()
        graphics.solve()                 # Solve TDSE *with* visualization


if __name__ == '__main__':
    exciton_tdse(batch_mode=False)