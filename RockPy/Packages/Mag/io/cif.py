import RockPy
import RockPy.core.ftype
import pandas as pd
import numpy as np

import os


class Cif(RockPy.core.ftype.Ftype):

    def __init__(self, dfile, snames=None, reload=False):
        super().__init__(dfile, snames=snames, dialect=None, reload=reload)

    def read_file(self):
        """
        The sample format is illustrated by the following fragment:
        00000000011111111112222222222333333333344444444445555555555666666666677777777778
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        erb  1.0A    Sample just above tuff
          113.0 291.0  63.0  43.0  46.0   1.0
        NRM     41.2  49.7  91.4  41.0 3.44E-05   5.5 184.1 -13.1  0.0289  0.0270  0.0468
        TT 150  46.7  41.3  84.3  33.7 1.79E-05   7.5 189.4 -20.9  0.0188  0.0130  0.0228
        TT 225  55.6  36.8  84.5  25.5 1.44E-05   4.0 197.8 -23.3  0.0193  0.0252  0.0171

        In the first line the first four characters are the locality id, the next 9 the sample id, and the remainder
        (to 255) is a sample comment.

        In the second line, the first character is ignored, the next 6 comprise the stratigraphic level
        (usually in meters). The remaining fields are all the same format: first character ignored (should be a blank
        space) and then 5 characters used. These are the core strike, core dip, bedding strike, bedding dip, and core
        volume or mass. Conventions are discussed below. CIT format can include fold axis and plunge, which at present
        is unused.

        The following lines are in the order the demagnetizations were carried out. The first 2 characters (3 for NRM
        only) is the demag type (AF for alternating field, TT for thermal, CH for chemical, etc.), the next 4 (3 for
        NRM) is the demag level (°C for thermal, mT for alternating field, etc.), the next 6 (first blank for all the
        following fields) for geographic ("in situ") declination of the sample's magnetic vector, next 6 for geographic
        inclination, next 6 for stratigraphic declination, next 6 for stratigraphic inclination, next 9 for normalized
        intensity (emu/cm^3; multiply by the core volume/mass to get the actual measured core intensity), next 6 for
        measurement error angle, next 6 for core plate declination, next 6 for core plate inclination, and the final
        three fields of 8 each are the standard deviations of the measurement in the core's x, y, and z coordinates
        in 10^5 emu. NB in 2003, it appears the CIT format is actually using three final fields of 9 characters, not 8.

        Presently only the sample id line, the second line, and the first ten fields (to core inclination but
        excepting the error angle) of the demag lines are used in PaleoMag. Except for the stratigraphic level,
        info on the second line is only displayed in the info window or used in the "Headers..." command. A possibility
        exists that future versions will plot Zijder plots with the measurement uncertainties.

        Returns
        -------

        """
        data = pd.DataFrame(
            columns=['mtype', 'level', 'geodec', 'geoinc', 'stratdec', 'stratinc', 'intensity', 'ang_err',
                     'plate_dec', 'plate_inc', 'std_x', 'std_y', 'std_z', 'user', 'date', 'time'])
        with open(self.dfile) as f:
            raw_data = f.readlines()

        self.raw_header_rows = []

        for idx, row in enumerate(raw_data):
            if len(row) == 115:
                num_index = min(i for i, v in enumerate(row) if v.isnumeric())

                # shift index for direction to be displayed in mtype
                if row.startswith('UAFX'):
                    num_index += 1

                data.loc[idx, 'mtype'] = row[:num_index].rstrip()

                if row.startswith('NRM'):
                    data.loc[idx, 'level'] = 0
                else:
                    data.loc[idx, 'level'] = int(row[num_index:6])

                # other columsn are separate by whitespace -> split(' ')
                values = [i for i in row[6:].split(' ') if i]

                # set string variables
                data.loc[idx, ['user', 'date', 'time']] = values[11:14]

                # set float variables
                data.loc[idx, data.columns[range(2, 13, 1)]] = np.array(values[:11]).astype(float)

            else:
                self.raw_header_rows.append(row)
        return data


if __name__ == '__main__':
    cif1 = Cif('/Users/mike/Dropbox/science/harvard/2G_data/mike/MIL/ARM(3000,20)/MIL11').data
    line0 = cif1.iloc[0].values
    print(line0[1] == 2)
    comp = ['ARM', 2, 331.3, -52.9, 331.3, -52.9, 4.13E-06, 000.4, 340.0, 32.0, 0.002359, 0.002268,
            0.000730, 'lancaste', '2019-07-18', '15:21:16']

    for i, v in enumerate(comp):
        print(i, v, line0[i], v == line0[i])
        # print(i, type(v), type(comp[i]), v==comp[i])
