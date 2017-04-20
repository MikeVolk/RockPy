import RockPy
from RockPy.core.ftype import Ftype
from RockPy.core import pandas_tools as pdt
import pandas as pd
import numpy as np

import os

class tdt(Ftype):

    pint_treatment_codes = ('LT-NO', 'LT-T-Z', 'LT-T-I', 'LT-PTRM-I', 'LT-PTRM-MD', 'LT-PTRM-Z')

    table = {'tdt':['NRM', '0', '1', '2', '3', '4']}

    def __init__(self, dfile, snames=None, volume= 1, dialect=None):
        super().__init__(dfile, snames=snames, dialect='tdt')

        self.volume = volume

        xyz = ['magn_x', 'magn_y', 'magn_z']

        data = pd.read_csv(dfile, delimiter='\t',
                       names=['specimen', 'step', 'magn_moment', 'dec', 'inc',], comment='#', skiprows=2,
                           dtype={'specimen':str, 'step':float, 'magn_moment':float, 'dec':float, 'inc':float})

        if snames:
            snames = RockPy.to_tuple(snames)
            data = data[np.in1d(data['specimen'], snames)]
            data = data.reset_index(drop=True)

        data['LT_code'] = [self.lookup_lab_treatment_code(i) for i in data['step']]

        data = pdt.DIM2XYZ(data,
                           colI='inc', colD='dec', colM='magn_moment',
                           colX='magn_x', colY='magn_y', colZ='magn_z')
        data['level'] = np.round(data['step'])
        data['step'] = data['level']
        self.add_tj_column(data)
        self.data = data

    @staticmethod
    def add_tj_column(data):
        """
        calculates the tj values
        Parameters
        ----------
        data

        Returns
        -------

        """
        for i, code in data['LT_code'].iteritems():
            if code in ('LT-PTRM-I', 'LT-PTRM-MD', 'LT-PTRM-Z'):
                tj = data[data['level'].index<i]['level'].max()
            else:
                tj = data[data['level'].index<=i]['level'].max()
            if np.isnan(tj):
                tj = 20

            data.loc[i,'tj'] = tj

    def lookup_lab_treatment_code(self, item):
        '''
        looks up the treeatment code for an item. This is done to be compatible with MagIC

        Notes
        -----
        Every dialect has to get its own if clause #todo maybe implement dialect_XXX function that adds a new column
        to the DataFrame

        Parameters
        ----------
        item

        Returns
        -------

        '''

        if self.dialect == 'tdt':
            split = str(item).split('.')
            idx = self.table[self.dialect].index(split[1])
            out = self.pint_treatment_codes[idx]

        return out

if __name__ == '__main__':
    RockPy.log.setLevel('DEBUG')
    f = os.path.join(RockPy.test_data_path, 'TT format', '187A.tdt')
    test = tdt(dfile=f)
    # print(test.data)
    s = RockPy.Sample(name='ET2_187A')
    m = s.add_measurement(mtype='paleointensity', ftype='tdt', fpath=f)
    # print(m.banc(vmin=150, vmax=300, blab=50))
    # print(m.sigma_banc(vmin=150, vmax=300, blab=50))
    print(m.fvds(vmin=150, vmax=300, blab=50))
    # print(m.results)



