import os
from typing import List
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter

figure_style = {
    # tight layout used for better visualisation
    'tight_layout': [0.05, 0.05, 0.95, 0.95],
    # xlimits used for the visualisation of non-expectation values
    'xlimits': (None, None),
    # ylimits used for the visualisation of non-expecation values
    'ylimits': (None, None),
    # format string for xticks
    'xtick_format': "%.0f",
    # format string for yticks
    'ytick_format': "%.1f",
    # scaling factor used in QuantDisplace2
    'scaling': 1
}

def scaling():
    if "FIGURE_SCALING" in os.environ:
        return float(os.environ["FIGURE_SCALING"])
    else:
        return 1

def set_scaling(scale: float):
    os.environ["FIGURE_SCALING"] = str(scale)

def set_axis_limits(axes: List[Axes], style: dict, direction: str):
    """
    Set new axes limits based on the 'xlimits' or 'ylimits' key given in 
    the style dictionary for all axis elements passed in the
    collection of axis elements.

    Parameters
    ------------
        axes: List[matplotlib.Axes]
            Collection of matplotlib.Axes instances
        style: dict
            Style dictionary containing 'xlim' keyword
        direction: str
            Can either be 'x' or 'y'
    """
    assert(direction in ['x', 'y'], "Direction must be 'x' or 'y'")
    ax_min, ax_max = style['xlimits'] if direction == 'x' else style['ylimits']
    
    for ax in axes:
        if direction == 'x':
            ax.set_xlim(ax_min, ax_max)
        else:
            ax.set_ylim(ax_min, ax_max)

def axis_tick_formatting(axes: List[Axes], style: dict, direction: str):
    """
    Set the string formatting of the axis ticks.
    """
    assert(direction in ['x', 'y'], "Direction must be 'x' or 'y'")

    format, attr = (style['xtick_format'], 'xaxis') if direction == 'x' else (style['ytick_format'], 'yaxis')

    for ax in axes: getattr(ax, attr).set_major_formatter(FormatStrFormatter(format))

def apply_styles(axes: List[Axes], style: dict):
    style_keys = style.keys()

    if "xlimits" in style_keys:
        if not (None, None) ==  style["xlimits"]:
            set_axis_limits(axes, style, 'x')

    if "ylimits" in style_keys:
        if not (None, None) == style["ylimits"]:
            set_axis_limits(axes, style, 'y')

    if "xtick_format" in style_keys:
        if style["xtick_format"] is not None:
            axis_tick_formatting(axes, style, 'x')

    if "ytick_format" in style_keys:
        if style["ytick_format"] is not None:
            axis_tick_formatting(axes, style, 'y')

    if "scaling" in style_keys:
        if style["scaling"] is not None:
            set_scaling(style['scaling'])