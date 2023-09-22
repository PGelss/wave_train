from wave_train.hamilton.bath_map_1 import Bath_Map_1
from wave_train.dynamics.tdse import TDSE
from wave_train.io.load import Load
from wave_train.io.logging import TeeLogger
from os.path import basename, splitext
import numpy as np
import matplotlib.pyplot as plt

def bath_map_1_tdse(batch_mode):
    # Detect name of this script file (without extension)
    base_name   = basename(__file__)
    my_file     = splitext(base_name)[0]

    # logging instance: will be initialized with 
    # class for logging to both console and logfile
    logger = None
    if not batch_mode:
        logger = TeeLogger(log_file=my_file + ".log")

    # Set up the excitonic Hamiltonian for a chain
    hamilton = Bath_Map_1(
        n_site=50,                       # number of sites
        eta = 0.5,                       # coupling between TLS and first bath site
        s = 1,                           # type of spectral density function: s<1 sub-ohmic, s=1 ohmic, s>1 super-ohmic
        omega_c = 10,                    # cut-off frequency of spectral density function
        omega_0 = 1                      # eigenfrequency of the TLS
    )

    # Set up TT representation of the Hamiltonian
    hamilton.get_TT(
        n_basis=2,                       # size of electronic basis set
        qtt=False                        # using quantized TT format
    )

    # Set up TDSE solver
    dynamics = TDSE(
        hamilton=hamilton,               # choice of Hamiltonian, see above
        num_steps=10,                    # number of main time steps
        step_size=0.1,                   # size of main time steps
        sub_steps=1,                     # number of sub steps
        solver='vp',                     # can be 'se' (symmetrized Euler) or 'sm' (Strang-Marchuk splitting) or ...
        normalize=0,                     # whether|how to normalize the solution, can be 0|2
        max_rank=5,                      # max rank of solution
        repeats=15,                      # number of sweeps (implicit ODE solvers only!)
        threshold=1e-8,                  # threshold in ALS decomposition
        save_file=my_file+'.pic',        # if not None, generated data will be saved to this file
        load_file=None,                  # if not None, reference data will be loaded from this file
        compare=None                     # How to do the comparison with reference data
    )

    # Set up initial state
    dynamics.fundamental(list(np.eye(hamilton.n_site)[0,:]))               # fundamental excitation near center of chain
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
    bath_map_1_tdse(batch_mode=False)


dynamics = Load(
    file_name='tdse_1',
    file_type = 'pic'
)
#print(dynamics.qu_numbr)
plt.figure()
for i in range(1,100,10):
    plt.plot(dynamics.qu_numbr[:,i])
plt.show()

#print(dynamics.qu_sig_1)


