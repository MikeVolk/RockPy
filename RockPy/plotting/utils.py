import os
import RockPy
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors as mcol, pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple

from RockPy.tools.compute import convert_to_equal_area
from RockPy.core.utils import maintain_n3_shape
from scipy.interpolate import griddata

def generate_plots(n=3, xsize=5., ysize=5., columns=None, tight_layout=True):
    """
    Generates a number of subplots that are organized in a way to fit on a landscape plot.

    Parameter
    ---------
        n: int
            number of plots to be generated
        xsize: float
            size along x for each plot
        ysize: float
            size along y for each plot
        rows: tries to fit the plots in as many rows
        tight_layout: bool
            using tight_layout (True) or not (False)

    Returns
    -------
       fig matplotlib figure instance
    """
    if columns:
        b = columns
        a = np.ceil(1. * n / b).astype(int)
    else:
        a = np.floor(n ** 0.5).astype(int)
        b = np.ceil(1. * n / a).astype(int)

    fig = plt.figure(figsize=(xsize * b, ysize * a))

    axes = []

    for i in range(1, n + 1):
        ax = fig.add_subplot(a, b, i)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True
        axes.append(ax)

    if tight_layout:
        fig.set_tight_layout(tight_layout)
    return fig, axes


""" AXIS """

def force_aspect(ax=None, aspect=1):
    """
    Changes the aspect of an axes to be `aspect`. Not by data
    Parameters
    ----------
    ax: matplotlib.axes
    aspect: float
    """

    if ax is None:
        ax = plt.gca()
    # aspect is width/height
    scale_str = ax.get_yaxis().get_scale()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if scale_str == 'linear':
        asp = abs((xmax - xmin) / (ymax - ymin)) / aspect
    elif scale_str == 'log':
        asp = abs((np.log10(xmax) - np.log10(xmin)) /
                  (np.log10(ymax) - np.log10(ymin))) / aspect
    ax.set_aspect(asp, adjustable='box')


def get_unique_axis(fig: plt.Figure):
    """
    Returns the unique (untwinned) axes for a figure. Uniqueness is determined by the extent
    of the position of the axes.

    Parameters
    ----------
    fig: matplotlib.figure

    Returns
    -------
        array(axes)
    """
    axes_positions = np.empty((1, 4))
    unique_axes = []

    for a in fig.axes:
        pos = np.array(a.get_position().extents)
        if not (axes_positions == pos).all(1).any():
            axes_positions = np.append(axes_positions, [pos], axis=0)
            unique_axes.append(a)

    return unique_axes


def add_twiny(label, ax=None, conversion=75.34):
    """
    Adds a second x axis on the top with a different scaling and label

    Parameters
    ----------
    ax: matplotlib.axes
    label: str  label for upper x axis

    Returns
    -------
    ax

    Notes
    -----
    Should be called at the end of a script, other wise tick-labels may be wrong #todo figure out how this can be fixed
    """

    if ax is None:
        ax = plt.gca()
    # twin ax
    ax2 = ax.twiny()

    # set new limits
    ax2.set_xlim(ax.get_xlim()[0] * conversion, ax.get_xlim()[1] * conversion)

    ax2.set_xlabel(label)

    return ax2


def max_zorder(ax):
    """

    Parameters
    ----------
    ax

    Returns
    -------
        maximum z order of any line in ax.
    """
    return max(_.zorder for _ in ax.get_children())


""" COLORS """

red_blue_colormap = mcol.LinearSegmentedColormap.from_list("MyCmapName", [
                                                           "b", "r"])


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))