import RockPy
import RockPy.core.ftype
import pandas as pd
import numpy as np

import os


class Jr6(RockPy.core.ftype.Ftype): #todo figure out how to import the data...
    pint_treatment_codes = ('LT-NO', 'LT-T-Z', 'LT-T-I', 'LT-PTRM-I', 'LT-PTRM-MD', 'LT-PTRM-Z')

    table = {'tdt': ['NRM', '0', '1', '2', '3', '4']}
    imported_files = dict()

    def __init__(self, dfile, snames=None, dialect=None, reload=False, volume=10 ** -5):
        super().__init__(dfile, snames=snames, dialect=dialect, reload = reload )
        self.volume = volume

        xyz = ['magn_x', 'magn_y', 'magn_z']
        # filter specimens
        if self.snames:
            self.data = self.data[np.in1d(self.data['specimen'], self.snames)]
            self.data = self.data.reset_index(drop=True)

        # divide by exponent
        self.data[xyz] = [v / 10 ** -self.data['exp'].iloc[i] for i, v in enumerate(self.data[xyz].values)]

        # unnormalize the volume
        self.data[xyz] *= self.volume

        # calculate Magnetic moment
        self.data['magn_moment'] = np.linalg.norm(self.data[xyz], axis=1)
        self.data['level'] = self.get_level()
        self.data['treat_temp'] = self.data['level'] + 273
        self.data['LT_code'] = [self.lookup_lab_treatment_code(i) for i in self.data['step']]
        self.data = self.data.drop('step', 1)
        self.data = self.data.drop('exp', 1)

    def get_level(self):

        if self.dialect == 'tdt':
            return [20 if i == 'NRM' else round(float(i.replace("T", ''))) for i in self.data['step']]

        if self.dialect == 'af':
            return [0 if i == 'NRM' else round(float(i.replace("A", ''))) for i in self.data['step']]


    def read_file(self):
        ''' 
        Method that does the actual reading of the whole file. All specimens are in the file
        '''

        data = pd.read_fwf(self.dfile, widths=(10, 8, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 2),
                           names=['specimen', 'step', 'magn_x', 'magn_y', 'magn_z', 'exp',
                                  'azimuth', 'dip',
                                  'foliation plane azimuth of dip', 'foliation plane dip',
                                  'lineation trend', 'lineation plunge',
                                  'P1', 'P2', 'P3', 'P4', 'Precision of measurement', 'CR-LF'
                                  ], usecols=[0, 1, 2, 3, 4, 5], comment='#')

        data['specimen'] = data['specimen'].astype(str)

        assert isinstance(data, pd.DataFrame)
        return data

    def lookup_lab_treatment_code(self, item):
        """
        looks up the treatment code for an item. This is done to be compatible with MagIC

        Notes
        -----
        Every dialect has to get its own if clause #todo maybe implement dialect_XXX function that adds a new column
        to the DataFrame

        Parameters
        ----------
        item

        Returns
        -------

        """
        out = None

        if self.dialect == 'tdt':
            if item.lower() == 'nrm':
                split = [0, item.upper()]
            else:
                split = item.split('.')

            idx = Jr6.table[self.dialect].index(split[1])
            out = Jr6.pint_treatment_codes[idx]

        if self.dialect == 'af':
            out = 'LT-AF-Z'

        return out


if __name__ == '__main__':
    j6 = Jr6(os.path.join(RockPy.test_data_path, 'TT-paleointensity.jr6'), dialect='tdt', snames=61)
    print(j6.data)
    print(j6.imported_files)
