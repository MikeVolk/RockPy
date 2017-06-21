import RockPy
import RockPy.core.ftype
import pandas as pd
import numpy as np

import os


class Jr6(RockPy.core.ftype.Ftype):
    pint_treatment_codes = ('LT-NO', 'LT-T-Z', 'LT-T-I', 'LT-PTRM-I', 'LT-PTRM-MD', 'LT-PTRM-Z')

    table = {'tdt': ['NRM', '0', '1', '2', '3', '4']}

    def __init__(self, dfile, snames=None, dialect=None, volume=10 ** -5):
        super().__init__(dfile, snames=snames, dialect=dialect)
        self.volume = volume

        xyz = ['magn_x', 'magn_y', 'magn_z']

        data = pd.read_fwf(dfile, widths=(10, 8, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 2),
                           names=['specimen', 'step', 'magn_x', 'magn_y', 'magn_z', 'exp',
                                  'azimuth', 'dip',
                                  'foliation plane azimuth of dip', 'foliation plane dip',
                                  'lineation trend', 'lineation plunge',
                                  'P1', 'P2', 'P3', 'P4', 'Precision of measurement', 'CR-LF'
                                  ], usecols=[0, 1, 2, 3, 4, 5], comment='#')

        if snames:
            snames = RockPy.to_tuple(snames)
            data = data[np.in1d(data['specimen'], snames)]
            data = data.reset_index(drop=True)

        assert isinstance(data, pd.DataFrame)

        # divide by exponent
        data[xyz] = [v / 10 ** -data['exp'].iloc[i] for i, v in enumerate(data[xyz].values)]

        # unnormalize the volume
        data[xyz] *= volume

        # calculate Magnetic moment
        data['magn_moment'] = np.linalg.norm(data[xyz], axis=1)
        data['level'] = [20 if i == 'NRM' else round(float(i.replace("T", ''))) for i in data['step']]
        data['treat_temp'] = data['level'] + 273
        data['LT_code'] = [self.lookup_lab_treatment_code(i) for i in data['step']]
        data = data.drop('step', 1)
        data = data.drop('exp', 1)
        self.data = data

    def read_file(self):
        pass

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

        return out


if __name__ == '__main__':
    j6 = Jr6(os.path.join(RockPy.test_data_path, 'TT-paleointensity.jr6'), dialect='tdt')
    print(j6.data)
