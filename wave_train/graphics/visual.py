import matplotlib
import matplotlib.pyplot as plt
from wave_train.graphics.animation import Animation

class Visual:
    """
    Class responsible for creating animated and styled output based
    on setup, configure and update services provided to the constructor.

    Invoking the solve routine will execute the setup, configure routines
    followed by calling the start routine of the dynamics instance.
    """
    def __init__(self, dynamics, setup, configure, update, animation_opts={}, style={}, backend="Qt5Agg"):
        """
        dynamics: Instance of TISE, TDSE, QCMD, CeoM
            When calling Visual.solve(), solve routine of dynamics instance
            will be called, followed by subsequent visualisation of the results
        setup: Callable
            Alters basic properties of matplotlib.pyplot figure object,
            such as size
        configure: Callable
            Advanced setup of matplotlib.pyplot figure object, such as setting
            grid dimensions, axes styling, ...
        update: Callable
            Update function rendering the new results of the dynamics
        animation_opts: dict
            Options used for setting up an Animation instance. Dictionary must
            provide the keys 'image_file', 'movie_file' and 'frame_rate'
        style: dict 
            Dictionary of information used to style a figure
        backend: string
            The backend used by matplotlib to render the plot
        """
        # must run before figure is created
        matplotlib.use(backend)

        self.figure         = plt.figure()
        self.dynamics       = dynamics
        self.setup          = setup
        self.configure      = configure
        self.update         = update
        self.animation_opts = animation_opts
        self.style          = style

    def solve(self):

        # Start the process of solving the dynamics dynamics
        self.dynamics.start_solve()

        # Setup basic properties of figure
        self.figure = self.setup(self.figure, self.style)
        self.figure = self.configure(self.figure, self.dynamics, style=self.style)

        movie_f = None
        image_f = None
        snapshots = False 

        aopt_keys = self.animation_opts.keys()

        if "movie_file" in aopt_keys:
            movie_f = self.animation_opts["movie_file"]

        if "image_file" in aopt_keys:
            image_f = self.animation_opts["image_file"]

        if "snapshots" in aopt_keys:
            snapshots = self.animation_opts["snapshots"] 

        # list holding the frame numbers 
        frames = range(self.dynamics.num_steps + 1)

        animation = Animation(self.dynamics, movie_f, image_f, snapshots, self.figure, self.update, 
                            frames=frames, pause=200, blit=False)

        animation.start()