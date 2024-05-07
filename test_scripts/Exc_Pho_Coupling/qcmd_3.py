from wave_train.hamilton.exc_pho_coupling import Exc_Pho_Coupling
from wave_train.dynamics.qcmd import QCMD
from wave_train.io.logging import TeeLogger
from os.path import basename, splitext

def coupled_qcmd(batch_mode):
    # Detect name of this script file (without extension)
    base_name   = basename(__file__)
    my_file     = splitext(base_name)[0]

    # logging instance: will be initialized with 
    # class for logging to both console and logfile
    logger = None
    if not batch_mode:
        logger = TeeLogger(log_file=my_file + ".log")

    # Set up the coupled exciton-phonon Hamiltonian for a chain
    hamilton = Exc_Pho_Coupling(
        n_site=41,                       # number of sites
        periodic=True,                   # periodic boundary conditions
        homogen=True,                    # Homogeneous chain/ring
        alpha=1e-1,                      # excitonic site energy
        beta=-1e-2,                      # coupling strength (NN)
        eta=0,                           # constant energy offset
        mass=1,                          # particle mass
        nu=1e-3,                         # position restraints
        omg=1e-3 * 2**(1/2),             # nearest neighbours
        chi=0e-4,                        # exciton-phonon tuning: localized
        rho=0e-4,                        # exciton-phonon tuning: non-symmetric
        sig=1.8e-4,                      # exciton-phonon tuning: symmetrized
        tau=0e-4                         # exciton-phonon coupling: pair distance
    )

    # Set up QCMD solver
    dynamics = QCMD(
        hamilton=hamilton,               # choice of Hamiltonian, see above
        num_steps=50,                    # number of main time steps
        step_size=0.01/hamilton.sig[0],  # size of main time steps
        sub_steps=100,                   # number of sub steps
        solver='sm',                     # can be 'lt' or 'sm' or 'pb' (pickaback)
        normalize=0,                     # whether|how to normalize the solution, can be 0|2
        save_file=my_file+'.pic',        # if not None, generated data will be saved to this file
        load_file=None,                  # if not None, reference data will be loaded from this file
        compare=None                     # How to do the comparison with reference data
    )

    # Set up initial state
    coeffs = hamilton.n_site * [1.0]     # Excitons completely delocalized
    noise = 0.1                          # Adding random noise to initial momenta
    dynamics.fundamental(coeffs, noise)

    # Batch mode
    if batch_mode:
        dynamics.solve()                 # Solve TDSE *without* visualization
    
    # Interactive mode: Setup animated visualization
    else:
        from wave_train.graphics.factory import VisualQCMD
        graphics = VisualQCMD(
            dynamics=dynamics,           # choice of dynamics (EoM), see above
            plot_type='QuantDisplace2',  # select your favorite plot style
            plot_expect=True,            # toggle plotting of expectation values
            figure_pos=(100, 50),        # specifying position (x,y) of upper left of figure [in pixels]
            figure_size=(1050, 450),     # specifying size (w,h) of figure [in pixels]
            image_file=my_file+'.png',   # if not None, image (last frame) will be written to this file
            movie_file=my_file+'.mp4',   # if not None, animation will be written to this file
            snapshots=False,             # save each snapshot
            frame_rate=1,                # frames per second in mp4 animation file
            plot_style={'scaling': 0.1}  # Scaling the classical coordinates
        ).create()
        graphics.solve()                 # Solve TDSE *with* visualization


if __name__ == '__main__':
    coupled_qcmd(batch_mode=False)