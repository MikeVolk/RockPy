import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy
import matplotlib.pyplot as plt
from RockPy.tools.plotting import MidpointNormalize


class VariForc(Ftype):
    header_ends = {'FORC function -1/2 ddM/(dHr dH) from corrected FORC measurements.': 'Data matrix',
                   'Backfield coercivity distribution f(x) = 1/2 dM(Hr,H)/dHr at Hr = -x and H = 0.':'Coercivity distribution on linear field scale (Hc, f(Hc), SE of f)'}

    def __init__(self, dfile, snames=None, dialect='processed', reload=False):
        """
        Args:
            dfile:
            snames:
            dialect:
            reload:
        """
        self.header = self.read_header(dfile)

        super().__init__(dfile, snames=snames, dialect=dialect, reload=reload)

    @classmethod
    def read_header(cls, dfile):
        """
        Args:
            dfile:
        """
        with open(dfile) as f:
            raw_header = f.readlines()

        header = {'mtype': raw_header[0].split('  ')[1].rstrip()}

        header['VariForc_version'] = raw_header[0].split('  ')[0].rstrip()
        header['data_start_idx'] = [i + 1 for i, v in enumerate(raw_header) if cls.header_ends[header['mtype']] in v][0]

        raw_header = np.array([i.rstrip() for i in raw_header[:header['data_start_idx']]])
        empty_idx = [i for i, v in enumerate(raw_header[:header['data_start_idx']]) if not v]

        for idx in empty_idx[:-1]:
            k = raw_header[idx + 1]
            v = raw_header[idx + 2].split(',')

            for i, x in enumerate(v):
                if x.startswith(' '):
                    x = x[1:]
                if all(
                    letter not in x.lower()
                    for letter in 'abcdfghijklmnopqrstuvwxyz'
                ):
                    v[i] = float(x)
                elif x.lower() == 'false':
                    v[i] = False
                elif x.lower() == 'true':
                    v[i] = True
                elif x.lower() == 'none':
                    v[i] = None

            if len(v) == 1:
                v = v[0]

            header[k] = v
        return header

    def read_file(self):
        # for k,v in self.header.items():
        #     print(k,v)

        if 'FORC function' in self.header['mtype']:
            hcvals = np.linspace(self.header['Horizontal range of grid points (Hcmin, Hcmax)'][0],
                                 self.header['Horizontal range of grid points (Hcmin, Hcmax)'][1],
                                 int(self.header['Grid dimensions (horizontal, vertical)'][0]))
            hbvals = np.linspace(self.header['Vertical range of grid points (Hbmin, Hbmax)'][0],
                                 self.header['Vertical range of grid points (Hbmin, Hbmax)'][1],
                                 int(self.header['Grid dimensions (horizontal, vertical)'][1]))

            data = np.loadtxt(self.dfile, skiprows=self.header['data_start_idx'], delimiter=',')[::-1]

            data = pd.DataFrame(index=hbvals, columns=hcvals, data=data)
            data.index.name = 'Hb'

        if self.dialect == 'backfield':
            data = pd.read_csv(self.dfile,
                               skiprows=66, index_col=0, names=('Hc', 'f(Hc)', 'SE of f'))
        return data

    def simple_plot(self, ax=None):

        """
        Args:
            ax:
        """
        if ax is None:
            ax = plt.gca()

        if 'FORC function' in self.header['mtype']:
            d = self.data
            x, y = (d.columns * 1000, d.index * 1000)

            cf = ax.imshow(d,
                           origin='lower',
                           extent=(min(x), max(x), min(y), max(y)),
                           #                       aspect = max(x)/max(y),
                           cmap=plt.get_cmap('RdBu_r'),
                           norm=MidpointNormalize(midpoint=0, vmin=d.min().min(), vmax=d.max().max())
                           )

            cb = plt.colorbar(cf, ax=ax)
            cb.ax.set_ylabel('Am$^2$/T$^2$')
            for i, v in enumerate(str(d.max().max())[2:]):
                if v != 0:
                    digit = i
                    break
            levels = self.get_levels()
            ax.contour(x, y, d, levels=levels, colors='k', linewidths=0.7)
            return ax

    def get_levels(self):
        """tryes to determin levels for plotting the forc contours :return:
        list
        """
        sigdig = 0

        mx = self.data.max().max()
        for i, stepsize in enumerate(np.logspace(10, -10, 21)):
            if i > 10:
                sigdig += 1
            if np.floor(mx / stepsize) != 0:
                break
        multiples = mx // stepsize
        levels = np.arange(0, np.round(multiples * stepsize + stepsize, sigdig), stepsize / 10)[1:]
        levels = np.round(levels, sigdig+1)

        while len(levels) >= 10:
            levels = levels[1::2]
        levels = np.concatenate([-levels[::-1], levels])

        return sorted(levels)


if __name__ == '__main__':
    d = VariForc(dialect='processed',
                 dfile='/Users/mike/Dropbox/science/_projects/magnetite_pressure/data/Chiton_Teeth/0/calculated/20141129_p0_FORC_VSM.frc_CorrectedMeasurements_VARIFORC.frc_CorrectedMeasurements_VARIFORC*_FORC_VARIFORC.txt').simple_plot()
