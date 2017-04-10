__author__ = 'mike'
from time import clock
import RockPy
import numpy as np
import pandas as pd
from RockPy.core import ftype
import os.path
from copy import deepcopy
# from RockPy3.core.utils import convert_time

class CryoMag(ftype.Ftype):
    imported_files = []
    pint_treatment_codes = ('LT-NO', 'LT-T-Z', 'LT-T-I', 'LT-PTRM-I', 'LT-PTRM-MD', 'LT-PTRM-Z')
    table = {'tdt':['NRM', 'TH', 'PT', 'CK', 'AC', 'TR']}

    def __init__(self, dfile, snames=None, dialect='tdt'):
        super().__init__(dfile=dfile, snames=snames, dialect=dialect)

    @property
    def _raw_data(self):
        out = CryoMag._clsdata[CryoMag._clsdata['dfile']==self.dfile]
        del out['dfile']
        return out

    @property
    def data(self):
        out =  self._raw_data[self._raw_data['mode']=='results']
        if self.snames is not None:
            out = out[np.in1d(out['name'], self.snames)]
        return out.reset_index(drop=True)

    def read_file(self):
        raw_data = pd.read_csv(self.dfile, delimiter='\t', skiprows=1, comment='#')
        raw_data = raw_data.rename(columns={"M": "magn_moment",
                                            "X [Am^2]": "magn_x", "Y [Am^2]": 'magn_y', "Z [Am^2]": 'magn_z',
                                            "type": 'LT_code', "name": "specimen", "step": "level"})
        raw_data['LT_code'] = [self.lookup_lab_treatment_code(i) for i in raw_data['LT_code']]

        raw_data['dfile'] = self.dfile
        CryoMag._clsdata = pd.concat([CryoMag._clsdata, raw_data])



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
            idx = CryoMag.table[self.dialect].index(item)
            out = CryoMag.pint_treatment_codes[idx]
        return out

if __name__ == '__main__':
    print(CryoMag(os.path.join(RockPy.test_data_path, 'TT-paleointensity.cryomag'), dialect='tdt').data)
    print(CryoMag(os.path.join(RockPy.test_data_path, 'TT-paleointensity.cryomag'), dialect='tdt').data)

