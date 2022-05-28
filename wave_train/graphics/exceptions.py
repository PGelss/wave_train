class PlotTypeNotSupported(Exception):
    def __init__(self, plot_type, dynamics):
        self.plot_type = plot_type 
        self.dynamics = dynamics

    def __str__(self):
        return f"Plot type {self.plot_type} is not supported for {self.dynamics} simulations"


class PlotTypeNotSuitable(Exception):
    def __init__(self, plot_type, dynamics, hamilton):
        self.plot_type = plot_type
        self.dynamics = dynamics
        self.hamilton = hamilton

    def __str__(self):
        return f"Plot type {self.plot_type} is not suitable for {self.dynamics} simulations of {self.hamilton}"


class InvalidDynamics(Exception):
    def __init__(self, factory, dynamics):
        self.factory = factory
        self.dynamics = dynamics
    
    def __str__(self):
        return f"VisualFactory{self.factory} does not support {self.dynamics} dynamics"


class InvalidHamiltonian(Exception):
    def __init__(self, factory, wrong, correct):
        self.factory = factory
        self.wrong = wrong
        self.correct = correct
    
    def __str__(self):
        return f"VisualFactory{self.factory} does not support {self.wrong} systems. " \
               f"It works only for {self.correct} systems."

