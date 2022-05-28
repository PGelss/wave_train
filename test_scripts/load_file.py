from wave_train.io.load import Load
from wave_train.graphics.factory import VisualTDSE, VisualTISE, VisualQCMD, VisualCEoM


def load_file():

    # Pic[kle] file from previous simulation
    data_file = 'Exciton/tdse_1.pic'
    my_file = data_file.split('.')[0]
    file_type = data_file.split('.')[-1]

    # Load data from pickle file
    dynamics = Load(my_file, file_type)

    if dynamics.name == "TISE":
        factory = VisualTISE

    # Check conservation of energy, norm, and RMSD (if available)
    elif dynamics.name in ['TDSE', 'QCMD']:
        dynamics.check_ENR()

        if dynamics.name == "TDSE":
            factory = VisualTDSE 
        else:
            factory = VisualQCMD

    # Redistribution of energy between kinetic and potential
    elif dynamics.name == 'CEoM':
        dynamics.check_EKP()
        factory = VisualCEoM

    # Redistribution of energy between quantum and classical subsystem
    elif dynamics.name == 'QCMD':
        dynamics.check_EQC()

    # Setup animated visualization
    graphics = factory(
        dynamics=dynamics,               # choice of "dynamics" (Load), see above
        plot_type='Populations',         # select your favorite plot style
        plot_expect=True,                # toggle plotting of expectation values
        figure_pos=(100, 50),            # specifying position (x,y) of upper left of figure [in pixels]
        figure_size=(1050, 450),         # specifying size (w,h) of figure [in pixels]
        image_file=my_file + '.png',     # if not None, image (last frame) will be written this file
        movie_file=my_file + '.mp4',     # if not None, animation will be written to this file
        frame_rate=1,                    # frames per second in mp4 animation file
        # pos_limits = (-25.0, 25.0)       # range of position axis (left, right)
        # mom_limits = (-0.05, 0.05)       # range of momentum axis (bottom, top)
        plot_style={}                    # additional plot style information
    ).create()
    graphics.solve()                     # Visualization only


if __name__ == '__main__':
    load_file()
