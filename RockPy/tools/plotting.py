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

colorpalettes = {
    'cat10': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf'],
    'cat20': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',
              '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
              '#17becf', '#9edae5'],
    'cat20b': ['#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31',
               '#bd9e39', '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194',
               '#ce6dbd', '#de9ed6'],
    'cat20c': ['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354',
               '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696',
               '#bdbdbd', '#d9d9d9'],
    'matlab': ['#0072bd', '#d95319', '#edb120', '#7e2f8e', '#77ac30', '#4dbeee', '#a2142f']}
ls = ['-', '--', '-.', ':'] * 100
marker = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'] * 100

# std figsize for AGU journals
figsize = np.array([3.74, 5.91]) * 1.3

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
        asp = abs((np.log10(xmax) - np.log10(xmin)) / (np.log10(ymax) - np.log10(ymin))) / aspect
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


def enumerate_figure(fig: plt.figure, positions=None, ignore=[], **kwargs):
    """
    Takes a figure object and places n) on each of the axes.

    Parameters
    ----------
    fig: matplotlib.figure
    positions: list of tuples
        list of tuples for the (x,y) positions of the text, relative to the axes
    kwargs: passed on to the text

    """
    axes = get_unique_axis(fig)

    if positions is None:
        positions = [(0.05, 0.85) for _, _ in enumerate(axes)]
    if np.array(positions).shape == (2,):
        positions = [positions for _, _ in enumerate(axes)]

    for i, ax in enumerate(axes):

        label = 'abcdefghijklmnopqrstuvwxyz'[i]

        if label in ignore or i in ignore:
            continue

        ax.text(positions[i][0], positions[i][1], '{:>s})'.format(label),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                bbox=kwargs.pop('bbox', dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0)),
                color=kwargs.pop('color', 'k'), **kwargs)


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


""" Stereonet and other projections"""


def setup_stereonet(ax=None, grid=True, rings=True, rtickwidth=1):
    """

    Parameters
    ----------
    ax
    grid
    rtickwidth

    Returns
    -------

    """
    if ax is None:
        ax = plt.subplot(111, projection='polar')

    ax.grid(grid)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_yticklabels([])  # Less radial ticks

    ## plot the grids
    # crosses
    for d in [0, 90, 180, 270]:
        d = convert_to_equal_area([np.ones(10) * d, np.linspace(0, 90, 10), np.ones(10)], intype='dim')
        ax.plot(np.radians(d[:, 0]), d[:, 1], 'k+')

<<<<<<< HEAD
    ticks = ax.set_rticks(d[:, 1])  # Less radial ticks
=======
    if rings:
        ticks = ax.set_rticks(d[:,1])  # Less radial ticks
    else:
        ticks = ax.set_rticks([])  # Less radial ticks

>>>>>>> development
    for t in ticks:
        t.set_alpha(0.5)

    # outer ticks
    for t in np.deg2rad(np.arange(0, 360, 5)):
        ax.plot([t, t], [1, 0.97], lw=rtickwidth, color="k", zorder=-1)

<<<<<<< HEAD
=======
    ax.set_thetagrids(angles=[0,90,180,270])
>>>>>>> development
    ax.set_rmax(1)
    return ax

def plot_stems(hkl, ymin=0, ymax=None, minI=0.5, ax=None, color=None):
    if ax is None:
        ax = plt.gca()

    hkl /= hkl.max()
    ymn, ymx = ax.get_ylim()

    for r in hkl.index:
        if hkl.loc[r]['Iobs'] >= minI:

            if ymax is None:
                y = hkl.loc[r]['Iobs']
            else:
                y = ymax

            ax.axvline(r, ymin=ymin / ymx, ymax=y / ymx, color=color, lw=0.7)


def plot_equal(xyz, ax=None, intype='xyz', setup_plot=True, **kwargs):
    """

    Parameters
    ----------
    xyz
    ax
    dim
    kwargs

    Returns
    -------
    ax
    down points
    up points
    """

    if ax is None:
        ax = plt.subplot(111, projection='polar')

    xyz = maintain_n3_shape(xyz)
    pol = convert_to_equal_area(xyz, intype=intype)

    if len(pol.shape) == 1:
        pol = pol.reshape((1, 3))

    neg_inc = pol[:, 2].astype(bool)
    down = pol[neg_inc]
    up = pol[~neg_inc]

    marker = kwargs.pop('marker', 'o')
    color = kwargs.pop('color', 'k')
    linecolor = kwargs.pop('linecolor', color)

    ls = kwargs.pop('ls', '')

    p1 = ax.plot(np.radians(down[:, 0]), down[:, 1], marker=marker, color=color, mfc='none', ls='', **kwargs)
    p2 = ax.plot(np.radians(up[:, 0]), up[:, 1], marker=marker, color=color, ls='', **kwargs)

    if ls:
        ax.plot(np.radians(pol[:, 0]), pol[:, 1], marker='', color=linecolor, ls=ls, **kwargs)

    if setup_plot:
        setup_stereonet(ax=ax)
    return ax


""" LINES """


def combined_label_legend(ax=None, pad=0.25, bbox_to_anchor=[1, 1],
                          add_handles=None, add_labels=None, add_sort=True,
                          **legend_opts):
    """
    Combines labels that are the same into one label

    Args:
        ax: matplotlib.axis
            default: None -> gca()
        pad: space between labels in a row
        bbox_to_anchor: location of legend
        **legend_opts: optional args passed to plt.legend
    """

    if ax is None:
        ax = plt.gca()

    h, l = ax.get_legend_handles_labels()

    if all(i is not None for i in [add_handles, add_labels]) and add_sort:
        h += add_handles
        l += add_labels

    labels = sorted(set(l))
    handles = [tuple(h[i] for i, l1 in enumerate(l) if l1 == l2) for n, l2 in enumerate(labels)]

    if all(i is not None for i in [add_handles, add_labels]) and not add_sort:
        h += add_handles
        l += add_labels

    mxlen = max([len(i) for i in handles])
    ax.legend(handles, labels, bbox_to_anchor=bbox_to_anchor,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=-1)},
              handletextpad=mxlen * pad,
              **legend_opts)


def log10_isolines(ax=None, angle=45):
    if ax is None:
        ax = plt.gca()

    ax.set_xscale('log')
    ax.set_yscale('log')

    xmn, xmx = ax.get_xlim()
    ymn, ymx = ax.get_ylim()

    # get line that halves the plot top let to bottom right
    s0 = (ymn - ymx) / (xmx - xmn)
    int0 = ymx - s0 * xmn

    # ax.plot(np.power(10, (np.array([20., -20.]) * s0 - int0)),
    # np.power(10, (np.array([-20., 20.]) * s0 - int0)), '-r')#, scaley=False, scalex=False)

    # plot iso lines
    for i, s in enumerate(np.arange(-20, 20, 1)):
        # for each power
        xnew = np.power(10, (np.array([-20., 20.]) - s / 2))
        ynew = np.power(10, (np.array([-20., 20.]) + s / 2))

        sn = (min(ynew) - max(ynew)) / (min(xnew) - max(xnew))
        intn = min(ynew) - sn * min(xnew)

        ax.plot(xnew, ynew, scaley=False, scalex=False, color='0.5', ls='--', zorder=0)

        rotation = ax.transData.transform_angles(np.array((angle,)),
                                                 np.array([1, 1]).reshape((1, 2)))[0]

        tx = (intn - int0) / (s0 - sn)
        ty = s0 * tx + int0

        ax.text(tx * 0.8, ty * 0.8,
                '10$^{%i}$' % s,
                verticalalignment='center', horizontalalignment='center', rotation=rotation,
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=0),
                color='k', clip_on=True, zorder=0.1)


def line_through_points(p1, p2, x_extent=None, ax=None, **kwargs):
    '''
    Creates a line through two points. First calculates the slope and intercept of those two points,
    then creates a line from x_extent[0] to x_extent[1]. The Line does not scale with the rest of the data.

    Parameters
    ----------
    p1: list(float, float)
        x,y of point 1
    p2: list(float, float)
        x,y of point 2
    x_extent: list(float, float)
        default: [-1000, 1000]
        x values to include
    ax: matplotlib.axes
        default: current axes
        axes object
    kwargs:
        passed to LineCollection

    '''
    if ax is None:
        ax = plt.gca()
    if x_extent is None:
        x_extent = [-1000, 1000]

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - slope * p1[0]

    if not ('linestyle' in kwargs or 'ls' in kwargs):
        kwargs = kwargs.update({'linestyle': '--'})

    if not ('color' in kwargs or 'c' in kwargs):
        kwargs.update({'color': '0.5'})

    if kwargs and ('ls' in kwargs):
        kwargs['linestyle'] = kwargs.pop('ls')

    line = np.array([[[x_extent[0], x_extent[0] * slope + intercept], [x_extent[1], x_extent[1] * slope + intercept]]])
    # Will not update limits
    line = LineCollection(line, **kwargs)
    ax.add_collection(line, autolim=False)


def add_zerolines(ax=None, **kwargs):
    """
    add x,y lines at x,y = 0

    Parameters
    ----------
    ax
    kwargs

    Returns
    -------

    """

    if ax is None:
        ax = plt.gca()

    ax.axhline(0, color=kwargs.pop('color', 'k'), zorder=kwargs.pop('zorder', 0), lw=kwargs.pop('lw', 1), **kwargs)
    ax.axvline(0, color=kwargs.pop('color', 'k'), zorder=kwargs.pop('zorder', 0), lw=kwargs.pop('lw', 1), **kwargs)


def make_spines_zero(a, xlabel='', ylabel=''):
    a.set_xlabel()
    a.set_ylabel('N/up ($10^{-9}$ Am$^2$)', rotation=-90)

    a.spines['left'].set_position('zero')
    a.spines['right'].set_color('none')
    a.spines['bottom'].set_position('zero')
    a.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    a.xaxis.set_ticks_position('bottom')
    a.yaxis.set_ticks_position('left')
    a.xaxis.set_label_coords(1.0, -0.005)
    a.yaxis.set_label_coords(1., 0.5)


def connect(p0, p1, ax=None, direction='up', arrow=False, shrink=2, **kwargs):
    """
    Connects 2 points by a horizontal line from p0[0]-p1[0] and a vertical line at p1[0] from p0[1] - p1[1].
    If an arrow is given, the direction can be chosen using direction = 'up', direction = 'down'

    Parameters
    ----------
    p0: (float, float)
        coordinates of point 1
    p1: (float, float)
        coordinates of point 1
    ax: matplotlib ax object
    direction: 'up' or 'down'
        default = 'up'
    arrow: bool

    shrink: int
        amount of pixels to shrink at the end of connection
    """
    x0, y0 = p0
    x1, y1 = p1
    if ax is None:
        ax = plt.gca()

    if arrow:
        arrowstyle: str = '->' if arrow else '-'
    else:
        arrowstyle = '-'

    if direction == 'down':
        ax.annotate("",
                    xy=(x1, y1), xycoords='data',
                    xytext=(x0, y1), textcoords='data',
                    arrowprops=dict(arrowstyle=arrowstyle, shrinkA=0, shrinkB=shrink, **kwargs),
                    )
        ax.annotate("",
                    xy=(x0, y0), xycoords='data',
                    xytext=(x0, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="-", shrinkA=0, shrinkB=shrink, **kwargs),
                    )

    elif direction == 'up':
        ax.annotate("",
                    xy=(x1, y1), xycoords='data',
                    xytext=(x1, y0), textcoords='data',
                    arrowprops=dict(arrowstyle=arrowstyle, shrinkA=0, shrinkB=shrink, **kwargs),
                    )

        ax.annotate("",
                    xy=(x1, y0), xycoords='data',
                    xytext=(x0, y0), textcoords='data',
                    arrowprops=dict(arrowstyle="-", shrinkA=shrink, shrinkB=0, **kwargs),
                    )
    else:
        RockPy.log.error('%s not a valid direction: choose either \'up\' or \'down\'' % direction)


def plot_square(x, y, d, center=True, center_label=None, ax=None, **plt_args):
    """
    plots a square with center = (x,y) and width = d

    Args:
        x: float
            x-coordinate of center
        y: float
            y-coordinate of center
        d: float
            width/height of square
        center_label: str, int
            default: None
            if given will be
        ax: axis object

    Returns:

    """
    if ax is None:
        ax = plt.gca()

    l, = ax.plot(x, y, 's', markersize=3, visible=False, label=f'{center_label}', **plt_args)

    if center:
        ax.plot(x, y, '+', markersize=3, color=l.get_color(), mew=1)

    ax.plot([x - d, x + d], [y - d, y - d], color=l.get_color(), lw=1)
    ax.plot([x - d, x + d], [y + d, y + d], color=l.get_color(), lw=1)
    ax.plot([x - d, x - d], [y - d, y + d], color=l.get_color(), lw=1)
    ax.plot([x + d, x + d], [y - d, y + d], color=l.get_color(), lw=1)

    return ax


def label_line(line, label, x, y, color='0.5', size=12):
    """Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinated
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    ax = line.axes
    text = ax.annotate(label, xy=(x, y), xytext=(-10, 0),
                       textcoords='offset points',
                       size=size, color=color,
                       horizontalalignment='left',
                       verticalalignment='bottom')

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text


def connect_ax_data(ax, **kwargs):
    """
    Connects all data points in ax with lines

    Parameters
    ----------
    ax: matplotlib.axes
    kwargs: dict
        passed to plot

    """
    x = []
    y = []
    for l in ax.lines:
        x.extend(l.get_xdata())
        y.extend(l.get_ydata())

    ax.plot(x, y, **kwargs)


""" COLORS """

red_blue_colormap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])


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


""" SPECIAL """


def plot_metamorphic_facies(ax=None, facies_list=None, text=None, **kwargs):
    """
    Plots metamorphic fascies into ax.
    Parameters
    ----------
    ax
    facies_list
    text
    kwargs

    Returns
    -------

    """
    xls = pd.ExcelFile(
        os.path.join(RockPy.installation_directory, 'tools', 'data', 'mtemorphic-facies_list-Gillen1982.xls'))

    if facies_list is None:
        facies_list = xls.sheet_names

    txtloc = {'limit metamorphism': (0, 0),
              'burial': (58, 86),
              'blueshist': (177, 685),
              'zeolite': (166, 265),
              'contact metamorphism': (525, 39),
              'greeschist': (0, 0),
              'amphibolite': (0, 0),
              'eclogite': (0, 0),
              'partial melting curve': (0, 0),
              'granulite': (0, 0)}
    txtrot = {'limit metamorphism': 0,
              'burial': 0,
              'blueshist': 0,
              'zeolite': 0,
              'contact metamorphism': 0,
              'greeschist': 0,
              'amphibolite': 0,
              'eclogite': 0,
              'partial melting curve': 0,
              'granulite': 0}

    for fascies in facies_list:
        # create deepcopy for same colors
        kwargscopy = deepcopy(kwargs)
        fontdictcopy = deepcopy(kwargs)

        if not fascies in xls.sheet_names:
            continue  # todo add warning

        data = pd.read_excel(xls, fascies).rolling(2).mean()

        ax.plot(data['T [C]'], data['P [MPa]'],
                color=kwargscopy.pop('color', 'k'),
                ls=kwargscopy.pop('ls', '--'),
                scalex=False, scaley=False,
                **kwargscopy)

        if text is not False and (text is None or fascies in text):
            ax.text(*txtloc[fascies], fascies,
                    rotation=txtrot[fascies],
                    verticalalignment='center', horizontalalignment='center',
                    bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0),
                    color='k', clip_on=True, **fontdictcopy)


class TernaryDiagram(object):

    @staticmethod
    def h(a):
        """
        calculates the height (y) at a given a (x)
        Parameters
        ----------
        a: float

        Returns
        -------
            float
        """
        if a == 0:
            return 0
        return np.sqrt(3) * a / 2

    def __init__(self, ax=None, extent=1, grid=True, **kwargs):
        """
        Ternary diagram for plotting data

        Parameters
        ----------
        ax: matplotlib.axes instance
            axes to be plotted on
        extent: default 1
        grid: bool
            plot default grid
            Note: for special grids call TernaryDiagram.grid method
        kwargs

        """
        # """

        if ax is None:
            ax = plt.gca()

        kwargs.setdefault('ls', '-')
        kwargs.setdefault('color', 'k')

        self.ax = ax
        self.extent = extent
        # plot the triangle
        self.ax.plot([0, 1], [0, 0], **kwargs)
        self.ax.plot([0, 0.5], [0, 1 / 2 * np.sqrt(3)], **kwargs)
        self.ax.plot([0.5, 1], [1 / 2 * np.sqrt(3), 0], **kwargs)

        if grid:
            self.grid()

    def grid(self, n=10, color='grey', ls='--', **kwargs):
        """
        Plots a triangular grid on the TernaryDiagram

        Parameters
        ----------
        n: int
            number of gridlines per axis
        color: matplotlib.color
            color of the lines
        ls: str
            linestyle of the grid
        """
        kwargs.setdefault('color', color)
        kwargs.setdefault('zorder', 1)
        kwargs.setdefault('ls', ls)
        kwargs.setdefault('lw', 0.7)

        for i in range(1, n + 1):
            a = self.extent
            self.ax.plot([(i / n) * a, a / 2 + (i / n) * (a / 2)], [0, self.h(a) - i * self.h((1 / n) * a)],
                         **kwargs)
            self.ax.plot([a - (i / n) * a, a - (a / 2 + (i / n) * (a / 2))], [0, self.h(a) - i * self.h((1 / n) * a)],
                         **kwargs)
            self.ax.plot([((i / n) * a) / 2, a - (i / n) / 2], [self.h(i / n * a), self.h(i / n * a)],
                         **kwargs)

    def transform(self, abc):
        """
        Transforms the coordinates for the three components a,b,c into cartesian coordinates (x,y).

        Parameters
        ----------
            abc: list
                list of (1,b,c) values

        Returns:
        --------
            xy: nd.array
                array of (x,y) values

        Consider an equilateral ternary plot where a = 100% is placed at (x,y) = (0,0) and b = 100% at (1,0).
        Then c = 100% is ($1/2,\sqrt(3)/2$), and the triple (a,b,c) is
        $\left({\frac {1}{2}}\cdot {\frac {2b+c}{a+b+c}},{\frac {\sqrt {3}}{2}}\cdot {\frac {c}{a+b+c}}\right)$
        """
        abc = np.array(abc)
        s = np.sum(abc, axis=1)

        abc = abc / s.reshape(len(s), 1)

        a = abc[:, 0]
        b = abc[:, 1]
        c = abc[:, 2]

        x = 1 / 2 * ((2 * b + c) / np.sum(abc, axis=1))
        y = np.sqrt(3) / 2 * (c / np.sum(abc, axis=1))

        xy = np.array(list(zip(x, y)))
        return xy

    def transform_2d(ab, ):
        '''
        transforms a 2D array into a 3D array using c = 1 - sqrt(a^2+b^2)

        Returns
        -------

        '''
        ab = np.array(ab)
        c = 1 - np.linalg.norm(ab, axis=1)
        abc = np.array([ab[:, 0], ab[:, 1], c]).T
        return abc

    def label_corner(self, corner, name, formula=None, **kwargs):
        """
        adds a name to one of the three corners:

        1: lower left
        2: lower right
        3: top

        :param name:
        :param corners: int
            1,2,3
        :param formula:
        :return:
        """

        coordinates = [[0, -.005], [1, -.005], [1 / 2, np.sqrt(3) / 2]]

        h_align = ['center', 'center', 'center']
        v_align = ['top', 'top', 'bottom']

        s = None
        if formula:
            s = formula
        if s:
            s = '\n'.join([s, name])
        else:
            s = name

        self.ax.text(*coordinates[corner], s=s, ha=h_align[corner], va=v_align[corner], **kwargs)

    def plot_abc(self, abc, **kwargs):
        '''
        Plots values in a,b,c normalized to 1 into the diagram

        Parameters
        ----------
        abc: list
            list of a,b,c values
        kwargs:
            passed on to plot function
        '''

        xy = self.transform(abc)
        self.ax.plot(xy[:, 0], xy[:, 1], **kwargs)

    def scatter_abc(self, abc, **kwargs):
        '''
        Plots values in a,b,c normalized to 1 into the diagram

        Parameters
        ----------
        abc: list
            list of a,b,c values
        kwargs:
            passed on to plot function
        '''
        kwargs.setdefault('zorder', 4)

        xy = self.transform(abc)
        self.ax.scatter(xy[:, 0], xy[:, 1], **kwargs)

    def plot_circle(self, corner, r, **kwargs):
        kwargs.setdefault('zorder', 4)

        if corner == 0:
            c = [[r, r * np.sin(i), r * np.cos(i)] for i in np.linspace(0, np.pi / 2)]
        elif corner == 1:
            c = [[r * np.sin(i), r, r * np.cos(i)] for i in np.linspace(0, np.pi / 2)]
        elif corner == 2:
            c = [[r * np.sin(i), r * np.cos(i), r] for i in np.linspace(0, np.pi / 2)]
        else:
            c = None
        self.plot_abc(c, **kwargs)

    def contour(self, abc, values, N=50, **kwargs):
        '''
        Places an interpolated contour plot on the diagram.

        Parameters
        ----------
        abc: array
            list of [a,b,c] coordinates for each value
        values: array
            list of values

        Notes
        -----
            abc[i] has to correspond to value[i]

        '''

        xy = self.transform(abc)
        xi = np.linspace(0, 1, 1000)
        yi = np.linspace(0, 1, 1000)

        zi = griddata(xy[:, 0], xy[:, 1], values, xi, yi, interp='linear')

        self.ax.contourf(xi, yi, zi, N, **kwargs)

    def show(self):
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        plt.show()
