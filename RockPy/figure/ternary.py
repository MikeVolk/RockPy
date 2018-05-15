import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata


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


if __name__ == '__main__':
    t = TernaryDiagram()
    t.grid(2, color='w')
    t.label_corner(0, 'C1', fontsize=14)

    t.label_corner(1, 'C2', fontsize=14)
    t.label_corner(2, 'C3', fontsize=14)

    t.plot_circle(0, 0.5, color='w')
    t.plot_circle(1, 0.5, color='w')
    t.plot_circle(2, 0.5, color='w')

    # make test data
    abc = []
    v = []
    for a in np.linspace(0, 1, 100):
        for b in np.linspace(0, 1 - a, 100):
            c = 1 - (a + b)
            z = sum([a * 0.5, b * 0.3, c * 0.1])
            abc.append([a, b, c])
            v.append(z)

            if a == 1:
                break

    t.contour(abc, v, 100)

    t.scatter_abc([np.random.rand(3) for i in np.linspace(0, 1)],
                  marker='o', c=['%.2f' % i for i in np.random.rand(50)])

    t.show()
