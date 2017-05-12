import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import os
import numpy as np
import io
from copy import deepcopy

class Vftb(Ftype):
    def __init__(self, dfile, snames=None, dialect=None):
        super().__init__(dfile, snames=snames, dialect=dialect)

        segment_idx = []
        header_idx = []
        with open(dfile) as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    weight = float(line.split('weight:')[1].replace('mg',''))*1e-6

                if 'set' in line.lower():
                    segment_idx.append(i)

                if 'field' in line:
                    self.header = line.replace('\n','').split('\t')
                    header_idx.append(i)

        self.units = [i.split('/')[1].strip() for i in self.header]
        conversion = [1e-3 if 'E-3' in unit else 1 for unit in self.units]

        self.mass = weight # mass in kg

        data = pd.read_csv(dfile, delimiter='\t', skiprows=[0,1]+segment_idx+header_idx,
                           names = ['B', 'M', 'T', 'time', 'std', 'sus'],
                           na_values = 'n/a')

        data['B'] /= 10000 # convert to tesla
        data['M'] /= self.mass # unnormalize
        data['sus'] /= self.mass # unnormalize

        data *= conversion

        self.data = data

        header_idx += [len(data)]
        self.segment_idx = [(header_idx[i]-min(header_idx), header_idx[i+1]-min(header_idx)) for i in range(len(header_idx)-1)]

    @property
    def segments(self):
        for i in self.segment_idx:
            yield self.data.iloc[i[0]:i[1]]

if __name__ == '__main__':
    vftb_file = os.path.join(RockPy.test_data_path, 'VFTB', 'hem4_20080702.rmp')

    data = Vftb(dfile=vftb_file)

    for s in data.segments:
        print(s)