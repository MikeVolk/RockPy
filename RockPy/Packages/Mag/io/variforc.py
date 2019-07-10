import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy


class VariForc(Ftype):
    header_ends = {'processed': 84,
                   'backfield':66}

    def __init__(self, dfile, snames=None, dialect='processed', reload=False):
        self.header = self.read_header(dfile, self.header_ends[dialect])

        super().__init__(dfile, snames=snames, dialect=dialect, reload=reload)

    @staticmethod
    def read_header(dfile, header_end):
        with open(dfile) as f:
            raw_header = f.readlines()[:header_end]

        raw_header = np.array([i.rstrip() for i in raw_header])
        empty_idx = [i for i, v in enumerate(raw_header) if not v]

        header = {}
        for idx in empty_idx[:-1]:
            k = raw_header[idx + 1]
            v = raw_header[idx+2].split(',')

            for i, x in enumerate(v):
                if x.startswith(' '):
                    x = x[1:]
                if not any(letter in x.lower() for letter in 'abcdfghijklmnopqrstuvwxyz'):
                    v[i] = float(x)
                elif x.lower() == 'false' or x.lower() == 'true':
                    v[i] = bool(x)
                elif x.lower() == 'none':
                    v[i] = None

            if len(v) == 1:
                v = v[0]

            header[k] = v

        return header

    def read_file(self):
        # for k,v in self.header.items():
        #     print(k,v)

        if self.dialect == 'processed':
            hcvals = np.arange(self.header['Horizontal range of grid points (Hcmin, Hcmax)'][0],
                               self.header['Horizontal range of grid points (Hcmin, Hcmax)'][1]+self.header['Grid mesh size'],
                               self.header['Grid mesh size'])
            hbvals = np.arange(self.header['Vertical range of grid points (Hbmin, Hbmax)'][0],
                               self.header['Vertical range of grid points (Hbmin, Hbmax)'][1],
                               self.header['Grid mesh size'])

            data = np.loadtxt(self.dfile, skiprows=84, delimiter=',')[::-1]

            data = pd.DataFrame(index=hbvals, columns=hcvals, data=data)
            data.index.name = 'Hb'

        if self.dialect == 'backfield':
            data = pd.read_csv(self.dfile,
                               skiprows=66, index_col=0, names=('Hc', 'f(Hc)', 'SE of f'))
        return data


if __name__ == '__main__':
    d = VariForc(dialect='backfield',
        dfile='/Users/mike/Dropbox/science/_projects/magnetite_pressure/data/Chiton_Teeth/A/calculated/20141129_A_FORC_VSM._CorrectedMeasurements_VARIFORC._CorrectedMeasurements_VARIFORC*_Backfield_Linear_VARIFORC.txt')
    print(d.data)
