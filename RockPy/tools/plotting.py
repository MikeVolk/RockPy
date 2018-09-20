# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcol

import RockPy


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


def enumerate_figure(fig, positions=None, **kwargs):
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
        ax.text(positions[i][0], positions[i][1], '{:>s})'.format('abcdefghijklmnopqrstuvwxyz'[i]),

                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0),
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

    # ax.plot(np.power(10, (np.array([20., -20.]) * s0 - int0)), np.power(10, (np.array([-20., 20.]) * s0 - int0)), '-r')#, scaley=False, scalex=False)



    # plot iso lines
    for i, s in enumerate(np.arange(-20, 20, 1)):
        # for each power
        xnew = np.power(10, (np.array([-20., 20.]) - s / 2))
        ynew = np.power(10, (np.array([-20., 20.]) + s / 2))

        sn = (min(ynew) - max(ynew)) / (min(xnew) - max(xnew))
        intn = min(ynew) - sn * min(xnew)

        ax.plot(xnew, ynew, scaley=False, scalex=False, color='0.5', ls='--')

        rotation = ax.transData.transform_angles(np.array((angle,)),
                                                 np.array([1, 1]).reshape((1, 2)))[0]

        tx = (intn - int0) / (s0 - sn)
        ty = s0 * tx + int0

        ax.text(tx * 0.8, ty * 0.8,
                '10$^{%i}$' % s,
                verticalalignment='center', horizontalalignment='center', rotation=rotation,
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=0),
                color='k', clip_on=True)


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


def forceAspect(ax=None, aspect=1):
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


def red_blue_colormap():
    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])
    return cm1


def connect_ax_data(ax, **kwargs):
    '''
    Connects the data of all lines in an axes

    Parameters
    ----------
    ax: matplotlib.axes
    kwargs: dict
        passed to plot

    '''
    x = []
    y = []
    for l in ax.lines:
        x.extend(l.get_xdata())
        y.extend(l.get_ydata())

    ax.plot(x, y, **kwargs)


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