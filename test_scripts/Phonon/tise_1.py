from wave_train.hamilton.phonon import Phonon
from wave_train.dynamics.tise import TISE
from os.path import basename, splitext


def phonon_tise(batch_mode):

    # Detect name of this script file (without extension)
    base_name = basename(__file__)
    my_file = splitext(base_name)[0]

    # Set up the phononic Hamiltonian for a chain
    hamilton = Phonon(
        n_site=6,                        # number of sites
        periodic=True,                   # periodic boundary conditions
        homogen=True,                    # homogeneous chain/ring
        mass=1,                          # particle masses
        nu=1e-3,                         # position restraints
        omg=1e-3 * 2**(1/2)              # nearest neighbours
    )

    # Set up TT representation of the Hamiltonian
    hamilton.get_TT(
        n_basis=8,                       # size of HO basis set
        qtt=False                        # using quantized TT format
    )

    # Set up TISE solver
    dynamics = TISE(
        hamilton=hamilton,               # choice of Hamiltonian, see above
        n_levels=10,                     # number of energy levels to be calculated
        solver='als',                    # choice of eigensolver for the full system
        eigen='eigs',                    # choice of eigensolver for the micro systems
        ranks=20,                        # rank of initial guess for ALS
        repeats=20,                      # number of sweeps in eigenproblem solver
        conv_eps=1e-8,                   # threshold for detecting convergence of the eigenvalue
        e_est=0,                         # estimation: eigenvalues closest to this number
        e_min=0.0065,                    # lower end of energy plot axis (if exact energies not available!)
        e_max=0.0090,                    # upper end of energy plot axis (if exact energies not available!)
        save_file=my_file+'.pic',        # if not None, generated data will be saved to this file
        load_file=None,                  # if not None, reference data will be loaded from this file
        compare=None                     # type of comparison with reference data
    )

    # Batch mode
    if batch_mode:
        dynamics.solve()                 # Solve TISE *without* visualization

    # Interactive mode: Setup animated visualization
    else:
        from wave_train.graphics.factory import VisualTISE
        graphics = VisualTISE(
            dynamics=dynamics,           # choice of dynamics (EoM), see above
            plot_type='QuantNumbers',    # select your favorite plot type
            plot_expect=True,            # toggle plotting of expectation values
            figure_pos=(100, 50),        # specifying position (x,y) of upper left of figure [in pixels]
            figure_size=(1050, 450),     # specifying size (w,h) of figure [in pixels]
            image_file=my_file+'.png',   # if not None, image (last frame) will be written this file
            movie_file=my_file+'.mp4',   # if not None, animation will be written to this file
            snapshots=False,             # save each snapshot
            frame_rate=1,                # Frames per second in mp4 animation file
            plot_style={}                # additional plot style information
        ).create()
        graphics.solve()                # Solve TISE *with* visualization


if __name__ == '__main__':
    phonon_tise(batch_mode=False)
