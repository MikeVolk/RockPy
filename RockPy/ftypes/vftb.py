import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import os
import numpy as np
import io
from copy import deepcopy

class Vftb(Ftype):
    def __init__(self, dfile, snames=None, dialect=None, reload=False):
        """
        Args:
            dfile:
            snames:
            dialect:
            reload:
        """
        self.mass, self.header, self.segment_idx, self.header_idx = self.read_header(dfile)
        super().__init__(dfile, snames=snames, dialect=dialect, reload = reload, header=self.header)

        # get the header_index for iterator
        self.header_idx += [len(self.data)]
        self.segment_idx = [(self.header_idx[i]-min(self.header_idx), self.header_idx[i+1]-min(self.header_idx)) for i in range(len(self.header_idx)-1)]

    @property
    def segments(self):
        for i in self.segment_idx:
            yield self.data.iloc[i[0]:i[1]]

    def read_header(self, dfile):
        """Reads the header of the VFTB file :param dfile: The location on your
        hard disk :type dfile: str

        Args:
            dfile:

        Returns:
            weight, segment_index, header_index:
        """

        segment_idx = []
        header_idx = []
        mass = None
        with open(dfile) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    mass = float(line.split('weight:')[1].replace('mg', '')) * 1e-6

                if 'set' in line.lower():
                    segment_idx.append(i)

                if 'field' in line:
                    header = line.replace('\n', '').split('\t')
                    header_idx.append(i)

        return mass, header, segment_idx, header_idx

    def read_file(self):

        self.units = [i.split('/')[1].strip() for i in self.header]
        conversion = [1e-3 if 'E-3' in unit else 1 for unit in self.units]

        data = pd.read_csv(self.dfile, delimiter='\t', skiprows=[0, 1] + self.segment_idx + self.header_idx,
                           names=['B', 'M', 'T', 'time', 'std', 'sus'],
                           na_values='n/a')

        data['B'] /= 10000  # convert to tesla
        data['M'] /= self.mass  # unnormalize
        data['sus'] /= self.mass  # unnormalize

        data *= conversion
        return data

if __name__ == '__main__':
    RockPy.welcome_message()
    s = RockPy.Sample('test')
    m = s.add_measurement(os.path.join(RockPy.test_data_path, 'vftb', 'hys_vftb.001'), mtype='hysteresis', ftype='vftb')