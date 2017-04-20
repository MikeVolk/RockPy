import RockPy
from RockPy.core.ftype import Ftype
from RockPy.core.utils import DIL2XYZ
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

        print(data[['dec','inc','magn_moment']])

        if snames:
            snames = RockPy.to_tuple(snames)
            data = data[np.in1d(data['specimen'], snames)]
            data = data.reset_index(drop=True)

        # data[xyz] = [v * 10 ** data['exp'].iloc[i] for i, v in enumerate(data[xyz].values)]
        # data[xyz] *= volume
        # data['magn_moment'] = np.linalg.norm(data[xyz], axis=1)
        # data['level'] = [20 if i == 'NRM' else round(float(i.replace("T", ''))) for i in data['step']]
        # data['treat_temp'] = data['level'] + 273
        data['LT_code'] = [self.lookup_lab_treatment_code(i) for i in data['step']]

        print(DIL2XYZ(data[['dec','inc','magn_moment']].values))
        # data = data.drop('step', 1)
        self.data = data


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
    f = os.path.join(RockPy.test_data_path, 'TT format', '187A.tdt')
    test = tdt(dfile=f)

    # s = RockPy.Sample(name='ET2_187A')
    # s.add_measurement(mtype='paleointensity', ftype='tdt', fpath=f)


