from wave_train.graphics.visual import Visual
from wave_train.graphics.services import (
    basic_figure_setup,
    configure_densitymat_basic,
    configure_densitymat_expect_tise,
    configure_phasespace_basic,
    configure_phasespace_expect_ceom,
    configure_phasespace_expect_tise,
    configure_positions2_expect_qcmd,
    configure_quant_displace2_expect_qcmd,
    configure_quant_numbers_basic,
    configure_quant_numbers_expect_tise,
    configure_populations_basic,
    configure_populations_expect_tise,
    configure_densitymat_expect_tdse,
    configure_populations_expect_tdse,
    configure_quant_numbers_expect_tdse,
    configure_phasespace_expect_tdse,
    configure_quant_numbers2_basic,
    configure_quant_numbers2_expect_tise,
    configure_quant_displace2_basic,
    configure_quant_displace2_expect_tise,
    configure_positions2_basic, 
    configure_positions2_expect_tise,
    configure_quant_displace2_expect_tdse,
    configure_positions2_expect_tdse,
    configure_quant_numbers2_expect_tdse,
    update_densitymat_basic,
    update_densitymat_expect_tise,
    update_phasespace_basic,
    update_phasespace_expect_ceom,
    update_phasespace_expect_tise,
    update_populations_basic,
    update_populations_expect_tise,
    update_positions2_expect_qcmd,
    update_quant_numbers_basic,
    update_quant_numbers_expect_tise,
    update_densitymat_expect_tdse,
    update_phasespace_expect_tdse,
    update_populations_expect_tdse,
    update_quant_numbers_expect_tdse,
    update_quant_numbers2_basic,
    update_quant_numbers2_expect_tise,
    update_quant_displace2_basic,
    update_quant_displace2_expect_tise,
    update_quant_displace2_expect_tdse,
    update_quant_displace2_expect_qcmd,
    update_positions2_basic,
    update_positions2_expect_tise,
    update_quant_numbers2_expect_tdse,
    update_positions2_expect_tdse,
)
from wave_train.graphics.style import figure_style
from wave_train.graphics.exceptions import InvalidHamiltonian, InvalidDynamics, \
    PlotTypeNotSupported, PlotTypeNotSuitable


class VisualFactory:
    """
    Interface class for creating instances of type 'class Visual' and providing
    the correct setup, start and update routines
    """
    __slots__ = [
        "plot_type",    # Visualisation type for configuration
        "dynamics",     # instance of one of the classes defined in wave_train.dynamics: TISE, TDSE, QCMD, CEoM
        "plot_expect",  # logical value whether to plot expectation values
        "figure_pos",   # display position of the figure
        "figure_size",  # size of plot figure
        "image_file",   # name of the file created from the final state of the animation
        "movie_file",   # name of movie file
        "frame_rate",   # number of frames per second
    ]

    def __init__(self, dynamics, plot_expect=True, figure_pos=(100, 50),
        figure_size=(1050, 450), image_file=None, movie_file=None, snapshots=False, frame_rate=1, style=figure_style):

        self.dynamics    = dynamics
        self.plot_expect = plot_expect
        self.figure_pos  = figure_pos
        self.figure_size = figure_size
        self.image_file  = image_file
        self.movie_file  = movie_file
        self.snapshots   = snapshots
        self.frame_rate  = frame_rate

        self.style = {
            'fig_size': self.figure_size,
            'fig_pos': self.figure_pos,
        }

        for key, value in style.items():
            self.style[key] = value

    def get_animation_opts(self):
        return {
            'image_file': self.image_file,
            'movie_file': self.movie_file,
            'frame_rate': self.frame_rate,
            'snapshots': self.snapshots
        }

    def create(self) -> Visual:
        raise NotImplementedError

class VisualTISE(VisualFactory):
    for_quantum = ["QuantNumbers", "Populations", "DensityMat"]
    for_classical = for_quantum + ["PhaseSpace"]
    for_bipartite = ["QuantNumbers2", "QuantDisplace2", "Positions2"]

    configure_callbacks = {
        "QuantNumbers": configure_quant_numbers_basic,
        "QuantNumbersExpect": configure_quant_numbers_expect_tise,
        "Populations": configure_populations_basic,
        "PopulationsExpect": configure_populations_expect_tise,
        "DensityMat": configure_densitymat_basic,
        "DensityMatExpect": configure_densitymat_expect_tise,
        "PhaseSpace": configure_phasespace_basic,
        "PhaseSpaceExpect": configure_phasespace_expect_tise,
        "QuantNumbers2": configure_quant_numbers2_basic,
        "QuantNumbers2Expect": configure_quant_numbers2_expect_tise,
        "QuantDisplace2": configure_quant_displace2_basic,
        "QuantDisplace2Expect": configure_quant_displace2_expect_tise,
        "Positions2": configure_positions2_basic,
        "Positions2Expect": configure_positions2_expect_tise
    }

    update_callbacks = {
        "QuantNumbers": update_quant_numbers_basic,
        "QuantNumbersExpect": update_quant_numbers_expect_tise,
        "Populations": update_populations_basic,
        "PopulationsExpect": update_populations_expect_tise,
        "DensityMat": update_densitymat_basic,
        "DensityMatExpect": update_densitymat_expect_tise,
        "PhaseSpace": update_phasespace_basic,
        "PhaseSpaceExpect": update_phasespace_expect_tise,
        "QuantNumbers2": update_quant_numbers2_basic,
        "QuantNumbers2Expect": update_quant_numbers2_expect_tise,
        "QuantDisplace2": update_quant_displace2_basic,
        "QuantDisplace2Expect": update_quant_displace2_expect_tise,
        "Positions2": update_positions2_basic,
        "Positions2Expect": update_positions2_expect_tise
    }

    def __init__(self, dynamics,
                 plot_type, plot_expect=True,
                 figure_pos=(100, 50), figure_size=(1050, 450),
                 image_file=None, movie_file=None, snapshots=False,
                 frame_rate=1, plot_style=figure_style):

        # if not TISE dynamics, throw an error
        if not dynamics.name == "TISE":
            raise InvalidDynamics("TISE", dynamics.name)

        # if plot type not in any of the 3 categories, throw an error
        if not plot_type in self.for_quantum + self.for_classical + self.for_bipartite:
            raise PlotTypeNotSupported(plot_type, dynamics.name)

        # if plot type not suitable for bipartite visualization, throw an error
        if dynamics.hamilton.bipartite:
            if plot_type not in self.for_bipartite:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        # if plot type not suitable for classical visualization, throw an error
        elif dynamics.hamilton.classical:
            if plot_type not in self.for_classical:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        # if plot type not suitable for quantum visualization, throw an error
        else:
            if plot_type not in self.for_quantum:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        self.plot_type = plot_type if not plot_expect else plot_type + "Expect"

        super().__init__(dynamics, plot_expect, figure_pos, figure_size,
                         image_file, movie_file, snapshots, frame_rate=frame_rate, style=plot_style)

    def create(self) -> Visual:
        """
        Injector function creating the Visual class with the correct
        services. Services are derived based on the provided plot type.

        Returns
        -----------
        An instance of class Visual 
        """
        setup_service       = basic_figure_setup 
        configure_service   = self.configure_callbacks[self.plot_type]
        update_service      = self.update_callbacks[self.plot_type]

        # animation options used for setting up the Animation instance
        animation_opts = self.get_animation_opts()

        client = Visual(self.dynamics, setup_service, configure_service, update_service, animation_opts, self.style)
        return client

class VisualTDSE(VisualFactory):
    for_quantum = ["QuantNumbers", "Populations", "DensityMat"]
    for_classical = for_quantum + ["PhaseSpace"]
    for_bipartite = ["QuantNumbers2", "QuantDisplace2", "Positions2"]

    configure_callbacks = {
        "QuantNumbers": configure_quant_numbers_basic,
        "QuantNumbersExpect": configure_quant_numbers_expect_tdse,
        "Populations": configure_populations_basic,
        "PopulationsExpect": configure_populations_expect_tdse,
        "DensityMat": configure_densitymat_basic,
        "DensityMatExpect": configure_densitymat_expect_tdse,
        "PhaseSpace": configure_phasespace_basic,
        "PhaseSpaceExpect": configure_phasespace_expect_tdse,
        "QuantNumbers2": configure_quant_numbers2_basic,
        "QuantNumbers2Expect": configure_quant_numbers2_expect_tdse,
        "QuantDisplace2": configure_quant_displace2_basic,
        "QuantDisplace2Expect": configure_quant_displace2_expect_tdse,
        "Positions2": configure_positions2_basic,
        "Positions2Expect": configure_positions2_expect_tdse
    }

    update_callbacks = {
        "QuantNumbers": update_quant_numbers_basic,
        "QuantNumbersExpect": update_quant_numbers_expect_tdse,
        "Populations": update_populations_basic,
        "PopulationsExpect": update_populations_expect_tdse,
        "DensityMat": update_densitymat_basic,
        "DensityMatExpect": update_densitymat_expect_tdse,
        "PhaseSpace": update_phasespace_basic,
        "PhaseSpaceExpect": update_phasespace_expect_tdse,
        "QuantNumbers2": update_quant_numbers2_basic,
        "QuantNumbers2Expect": update_quant_numbers2_expect_tdse,
        "QuantDisplace2": update_quant_displace2_basic,
        "QuantDisplace2Expect": update_quant_displace2_expect_tdse,
        "Positions2": update_positions2_basic,
        "Positions2Expect": update_positions2_expect_tdse
    }

    def __init__(self, dynamics,
                 plot_type, plot_expect=True,
                 figure_pos=(100, 50), figure_size=(1050, 450),
                 image_file=None, movie_file=None, snapshots=False,
                 frame_rate=1, plot_style=figure_style):

        # if not TDSE dynamics, throw an error
        if not dynamics.name == "TDSE":
            raise InvalidDynamics("TDSE", dynamics.name)

        # if plot type not in any of the 3 categories, throw an error
        if plot_type not in self.for_quantum + self.for_classical + self.for_bipartite:
            raise PlotTypeNotSupported(plot_type, dynamics.name)

        # if plot type not suitable for bipartite visualization, throw an error
        if dynamics.hamilton.bipartite:
            if plot_type not in self.for_bipartite:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        # if plot type not suitable for classical visualization, throw an error
        elif dynamics.hamilton.classical:
            if plot_type not in self.for_classical:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        # if plot type not suitable for quantum visualization, throw an error
        else:
            if plot_type not in self.for_quantum:
                raise PlotTypeNotSuitable(plot_type, dynamics.name, dynamics.hamilton.name)

        self.plot_type = plot_type if not plot_expect else plot_type + "Expect"

        super().__init__(dynamics, plot_expect, figure_pos, figure_size,
                         image_file, movie_file, snapshots, frame_rate=frame_rate, style=plot_style)

    def create(self) -> Visual:
        """
        Injector function creating the Visual class with the correct
        services. Services are derived based on the provided plot type.

        Returns
        -----------
        An instance of class Visual 
        """
        setup_service       = basic_figure_setup 
        configure_service   = self.configure_callbacks[self.plot_type]
        update_service      = self.update_callbacks[self.plot_type]

        # animation options used for setting up the Animation instance
        animation_opts = self.get_animation_opts()

        client = Visual(self.dynamics, setup_service, configure_service, update_service, animation_opts, style=self.style)
        return client

class VisualQCMD(VisualFactory):
    supported_plot_types = ["Positions2", "QuantDisplace2"]

    configure_callbacks = {
        "QuantDisplace2": configure_quant_displace2_basic,
        "QuantDisplace2Expect": configure_quant_displace2_expect_qcmd,
        "Positions2": configure_positions2_basic,
        "Positions2Expect": configure_positions2_expect_qcmd
    }

    update_callbacks = {
        "QuantDisplace2": update_quant_displace2_basic,
        "QuantDisplace2Expect": update_quant_displace2_expect_qcmd,
        "Positions2": update_positions2_basic,
        "Positions2Expect": update_positions2_expect_qcmd
    }

    def __init__(self, dynamics,
                 plot_type, plot_expect=True,
                 figure_pos=(100, 50), figure_size=(1050, 450),
                 image_file=None, movie_file=None, snapshots=False,
                 frame_rate=1, plot_style=figure_style):

        # if not QCMD dynamics, throw an error
        if not dynamics.name == "QCMD":
            raise InvalidDynamics("QCMD", dynamics.name)

        # if not for bipartite systems, throw an error
        if not dynamics.hamilton.bipartite:
            raise InvalidHamiltonian("QCMD", dynamics.hamilton.name, "bipartite")

        # if plot type not supported, throw an error
        if plot_type not in self.supported_plot_types:
            raise PlotTypeNotSupported(plot_type, dynamics.name)

        self.plot_type = plot_type if not plot_expect else plot_type + "Expect"

        super().__init__(dynamics, plot_expect, figure_pos, figure_size,
                         image_file, movie_file, snapshots, frame_rate=frame_rate, style=plot_style)

    def create(self):
        setup_service       = basic_figure_setup 
        configure_service   = self.configure_callbacks[self.plot_type]
        update_service      = self.update_callbacks[self.plot_type]

        # animation options used for setting up the Animation instance
        animation_opts = self.get_animation_opts()

        client = Visual(self.dynamics, setup_service, configure_service, update_service, animation_opts, self.style)
        return client

class VisualCEoM(VisualFactory):
    supported_plot_types = ["PhaseSpace"]
    
    configure_callbacks = {
        "PhaseSpace": configure_phasespace_basic,
        "PhaseSpaceExpect": configure_phasespace_expect_ceom 
    }

    update_callbacks = {
        "PhaseSpace": update_phasespace_basic,
        "PhaseSpaceExpect": update_phasespace_expect_ceom
    }

    def __init__(self, dynamics,
                 plot_type, plot_expect=True,
                 figure_pos=(100, 50), figure_size=(1050, 450),
                 image_file=None, movie_file=None, snapshots=False,
                 frame_rate=1, plot_style=figure_style):

        # if not CEoM dynamics, throw an error
        if not dynamics.name == "CEoM":
            raise InvalidDynamics("CEoM", dynamics.name)

        # if not for classical systems, throw an error
        if not dynamics.hamilton.classical:
            raise InvalidHamiltonian("CEoM", dynamics.hamilton.name, "classical")

        # if plot type not supported, throw an error
        if plot_type not in self.supported_plot_types:
            raise PlotTypeNotSupported(plot_type, dynamics.name)

        self.plot_type = plot_type if not plot_expect else plot_type + "Expect"

        super().__init__(dynamics, plot_expect, figure_pos, figure_size,
                         image_file, movie_file, snapshots, frame_rate=frame_rate, style=plot_style)

    def create(self) -> Visual:
        setup_service       = basic_figure_setup 
        configure_service   = self.configure_callbacks[self.plot_type]
        update_service      = self.update_callbacks[self.plot_type]

        # animation options used for setting up the Animation instance
        animation_opts = self.get_animation_opts()

        client = Visual(self.dynamics, setup_service, configure_service, update_service, animation_opts, self.style)
        return client