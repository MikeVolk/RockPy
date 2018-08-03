# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
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
        RockPy.log.error('%s not a valid direction: choose either \'up\' or \'down\'' % (direction))


def get_unique_axis(fig):
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
        list of tuples for the (x,y) positions of the text, realtive to the axes
    kwargs: passed on to the text

    """

    axes = get_unique_axis(fig)

    if positions is None:
        positions = [(0.05, 0.9) for i, ax in enumerate(axes)]
    if np.array(positions).shape == (2,):
        positions = [positions for i, ax in enumerate(axes)]

    for i, ax in enumerate(axes):
        ax.text(positions[i][0], positions[i][1], '{:>s})'.format('abcdefghijklmnopqrstuvwxyz'[i]),

                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0),
                color=kwargs.pop('color', 'k'), **kwargs)


def add_twiny(ax, label, conversion=75.34):
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
    Should be called at the end of a script, other wise ticklabels may be wrong #todo figure out how this can be fixed
    """
    # twin ax
    ax2 = ax.twiny()

    # set new limits
    ax2.set_xlim(ax.get_xlim()[0] * conversion, ax.get_xlim()[1] * conversion)

    ax2.set_xlabel(label)

    return ax2


def add_log10_isolines(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xscale('log')
    ax.set_yscale('log')

    xmn, xmx = ax.get_xlim()
    ymn, ymx = ax.get_ylim()

    # get line that halves the plot top to bottom
    s0 = (ymn - ymx) / (xmx - xmn)
    int0 = ymx - s0 * xmn
    # ax.plot(np.linspace(xmn,xmx), np.linspace(xmn,xmx)*s0+int0, color='r', ls='--')

    for i, s in enumerate(np.arange(-10, 10, 1)):
        xnew = np.power(10, (np.array([-20., 20.]) - s / 2))
        ynew = np.power(10, (np.array([-20., 20.]) + s / 2))

        sn = (min(ynew) - max(ynew)) / (min(xnew) - max(xnew))
        intn = min(ynew) - sn * min(xnew)

        # ax.plot(xnew, xnew*sn+intn, scaley=False, scalex=False, ls='--')

        ax.plot(xnew, ynew, scaley=False, scalex=False, color='0.5', ls='--')

        rotation = ax.transData.transform_angles(np.array((45,)),
                                                 np.array([1, 1]).reshape((1, 2)))[0]

        tx = (intn - int0) / (s0 - sn)
        ty = s0 * tx + int0
        ax.text(tx, ty,
                '10$^{%i}$' % s,
                verticalalignment='center', horizontalalignment='center', rotation=rotation,
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=0),
                color='k', clip_on=True)


def getvivible_data(line):
    '''
    takes a line and ax and returns the data that is actually visible
    Parameters
    ----------
    line

    Returns
    -------

    '''
