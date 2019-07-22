import RockPy
import RockPy.core.ftype
from RockPy.tools.pandas_tools import XYZ2DIM, DIM2XYZ, rotate
from datetime import datetime
import pandas as pd
import numpy as np
import pmagpy

import os


class Cif(RockPy.core.ftype.Ftype):
    datacolumns = ['mtype', 'level', 'geo_dec', 'geo_inc', 'strat_dec', 'strat_inc', 'intensity', 'ang_err',
                   'plate_dec', 'plate_inc', 'std_x', 'std_y', 'std_z', 'user', 'date', 'time']

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
            columns=['mtype', 'level', 'geo_dec', 'geo_inc', 'strat_dec', 'strat_inc', 'intensity', 'ang_err',
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

                # other columns are separate by whitespace -> split(' ')
                values = [i for i in row[6:].split(' ') if i]

                # set string variables
                data.loc[idx, ['user', 'date', 'time']] = values[11:14]

                # set float variables
                data.loc[idx, data.columns[range(2, 13, 1)]] = np.array(values[:11]).astype(float)

            else:
                self.raw_header_rows.append(row)
        return data

    @staticmethod
    def write_cif_line(series):

        timedate = series[0]
        series = series[1]

        mtype = series['mtype']
        level = series['level']

        columns = ['geo_dec', 'geo_inc', 'strat_dec', 'strat_inc', 'intensity', 'ang_err',
                   'plate_dec', 'plate_inc', 'std_x', 'std_y', 'std_z', 'user']
        formats = {'mtype': '{:<2}', 'level': '{:>4}',
                   'geo_dec': '{:>5.1f}', 'geo_inc': '{:>5.1f}', 'strat_dec': '{:>5.1f}', 'strat_inc': '{:>5.1f}',
                   'intensity': '{:.2E}', 'ang_err': '{:05.1f}', 'plate_dec': '{:>5.1f}', 'plate_inc': '{:>5.1f}',
                   'std_x': '{:>.6f}', 'std_y': '{:>.6f}', 'std_z': '{:>.6f}',
                   'user': '{:>7}', 'date': '{:>4}', 'time': '{:>4}'}

        if mtype == 'NRM':
            formats['mtype'] = "{:<3}"
            formats['level'] = '{:>3}'
            level = ''

        if mtype == "UAFX":
            formats['mtype'] = "{:<4}"
            formats['level'] = '{:>2}'

        start = formats['mtype'].format(mtype) + formats['level'].format(level) + ' '
        rest = ' '.join(formats[fmt].format(series[fmt]) for fmt in columns)
        date = timedate.strftime(' %Y-%m-%d')
        time = timedate.strftime(' %H:%M:%S')

        return start + rest + date + time + '\n'

    def __df_to_StringIO(self):
        '''
        import io

        output = io.StringIO()
        output.write('First line.\n')
        print('Second line.', file=output)

        # Retrieve file contents -- this will be
        # 'First line.\nSecond line.\n'
        contents = output.getvalue()

        # Close object and discard memory buffer --
        # .getvalue() will now raise an exception.
        output.close()
        Returns
        -------

        '''
        pass

    @classmethod
    def from_rapid(cls, files_or_folder,
                   sample_id, locality_id='',
                   core_strike=0., core_dip=0.,
                   bedding_strike=0., bedding_dip=0.,
                   core_volume_or_mass=1,
                   stratigraphic_level='', comment='', user='lancaster'):
        """
        reads Up/DOWN files or folder containing these and creates a cif-like object

        Parameters
        ----------
        files_or_folder

        Returns
        -------

        """
        if os.path.isdir(files_or_folder):
            files = [os.path.join(files_or_folder, f) for f in os.listdir(files_or_folder)]
        else:
            files = RockPy.to_tuple(files_or_folder)

        files = sorted([f for f in files if (f.endswith('UP') or f.endswith('DOWN'))])
        files = files[:20]
        # read all the files , create list of Dataframes
        raw_df = []
        for i, df in enumerate(files):
            print('reading file << {:>20} >> {:>4} of {:>4}'.format(os.path.basename(df), i, len(files)), end='\r')
            readdf = cls.read_UP_file(df, sample_id)

            if readdf is not None:
                raw_df.append(readdf)
                cls.imported_files[df] = readdf

        average_df = []
        for i, df in enumerate(raw_df):
            print('averaging file {:>4} of {:>4}'.format(i, len(raw_df)), end='\r')
            average_df.append(cls.return_mean_from_UP_file(df))
        if len(average_df) > 1:
            data = pd.concat(average_df)
        else:
            data = average_df[0]

        data = XYZ2DIM(data, colX='x', colY='y', colZ='z', colI='plate_inc', colD='plate_dec', colM='intensity')
        data = data.sort_index()
        data['user'] = user[:8]

        # data['z'] *= -1

        cls.correct_dec_inc(data, core_dip, core_strike,
                            colI='plate_inc', colD='plate_dec',
                            newD='geo_dec', newI='geo_inc')

        cls.correct_dec_inc(data, bedding_dip, bedding_strike,
                            colI='geo_inc', colD='geo_dec',
                            newI='strat_inc', newD='strat_dec')

        out = ['{:>4}{:>9}{}\n'.format(locality_id, sample_id, comment[:255]),
               ' {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}\n'.format(stratigraphic_level, core_strike, core_dip,
                                                               bedding_strike, bedding_dip, core_volume_or_mass),
               ]

        for row in data.iterrows():
            out.append(cls.write_cif_line(row))

        return out

    @classmethod
    def correct_dec_inc(cls, df, dip, strike, newI, newD, colD='D', colI='I'):
        DI = df[[colD, colI]]
        DI = DIM2XYZ(DI, colI=colI, colD=colD, colM=None)

        xyz = DI[['x', 'y', 'z']]

        if dip:
            xyz = rotate(xyz, axis='x', deg=dip)
        if strike:
            xyz = rotate(xyz, axis='z', deg=strike)
        corrected = XYZ2DIM(pd.DataFrame(columns=['x', 'y', 'z'], data=xyz, index=DI.index),
                            colI=newI, colD=newD)

        df[newI] = corrected[newI]
        df[newD] = corrected[newD]
        return df

    @staticmethod
    def return_mean_from_UP_file(df):
        """
        takes a DataFrame created from reading in an (UP?DOWN) file (see. read_UP_files) and returns a new dataframe
        where the average XYZ values of the holder was subtracted and the XYZ components have been averaged.
        Parameters
        ----------
        df

        Returns
        -------

        """
        means = {'S': None, 'H': None, 'Z': None}
        stdevs = {'S': None, 'H': None, 'Z': None}

        for mtype in ('S', 'Z', 'H'):
            d = df[df['MsmtType'] == mtype]

            mean = d.groupby(d.index).mean()
            std = d.groupby(d.index).std()

            means[mtype] = mean
            stdevs[mtype] = std

        out = pd.DataFrame(
            columns=['mtype', 'level', 'geo_dec', 'geo_inc', 'strat_dec', 'strat_inc', 'intensity', 'ang_err',
                     'plate_dec', 'plate_inc', 'std_x', 'std_y', 'std_z', 'user', 'date', 'time', 'sample'])

        for t in set(df.index):
            out.loc[t, 'mtype'] = df.iloc[0]['mtype']
            out.loc[t, 'level'] = df.iloc[0]['level']
            out.loc[t, 'sample'] = df.iloc[0]['Sample']
            out.loc[t, 'mtype'] = df.iloc[0]['mtype']

        out[['x', 'y', 'z']] = Cif.correct_holder(means['S'][['x', 'y', 'z']], means['H'][['x', 'y', 'z']])

        # calculate Dec/Inc standard deviations # todo calculate this properly i.e. alpha 95 or something

        # add standard deviations to df
        out.loc[t, 'std_x'] = stdevs['S'].loc[t, 'x']
        out.loc[t, 'std_y'] = stdevs['S'].loc[t, 'y']
        out.loc[t, 'std_z'] = stdevs['S'].loc[t, 'z']

        out.loc[t, 'ang_err'] = stdevs['S'].loc[t, 'D']

        return out

    @staticmethod
    def correct_holder(sample_means, holder_means):
        # subtract holder measurement
        corrected = sample_means - holder_means
        return corrected

    @staticmethod
    def read_UP_file(dfile, sample_id):
        with open(dfile) as f:
            f = f.readlines()

        f = [n.rstrip().replace(',', '|') for n in f]
        f = [n.split('|') for n in f]

        f[0].append('datetime')

        out = pd.DataFrame(columns=f[0], data=f[1:])

        if not sample_id in set(out['Sample']):
            RockPy.log.error('Could not find sample_id << {} >> in file << {} >.! '
                             'Please check correct spelling'.format(sample_id, os.path.basename(dfile)))
            return

        out = out[out['Sample'] == sample_id]

        out['datetime'] = pd.to_datetime(out['datetime'])

        out[['MsmtNum']] = out[['MsmtNum']].astype(int)
        out[['X', 'Y', 'Z']] = out[['X', 'Y', 'Z']].astype(float)
        out[['X', 'Y']] = out[['X', 'Y']]
        out[['Y']] *= -1

        out = out.rename(columns={"X": "X_", "Y": "Y_", "Z": "Z_"})
        # rotate measurements into sample coordinates
        if dfile.endswith('.UP'):
            for idx, v in out.iterrows():
                x, y, z = v[['X_', 'Y_', 'Z_']].values

                if v['MsmtNum'] == 1:
                    out.loc[idx, 'x'] = x
                    out.loc[idx, 'y'] = y
                if v['MsmtNum'] == 2:
                    out.loc[idx, 'x'] = -y
                    out.loc[idx, 'y'] = x
                if v['MsmtNum'] == 3:
                    out.loc[idx, 'x'] = -x
                    out.loc[idx, 'y'] = -y
                if v['MsmtNum'] == 4:
                    out.loc[idx, 'x'] = y
                    out.loc[idx, 'y'] = -x
                out.loc[idx, 'z'] = z

            out.loc[idx, 'M'] = np.linalg.norm([x, y, z])

        out = XYZ2DIM(out, colX='x', colY='y', colZ='z', colD='D', colI='I', colM='M')
        # out['D'] = (out['D'] + 90) % 360
        out = out.set_index('datetime')
        dfile = os.path.basename(dfile)
        out['dfile'] = dfile
        out['mtype'] = ''.join([n for n in dfile.split('.')[0] if not n.isnumeric()]).rstrip()
        out['level'] = ''.join([n for n in dfile.split('.')[0] if n.isnumeric() if n])
        out['level'] = [int(i) if i else 0 for i in out['level']]
        return out


if __name__ == '__main__':
    cif2 = Cif.from_rapid('/Users/mike/Dropbox/science/harvard/2G_data/mike/HAAL', sample_id='HAAL1d',
                          core_dip=90, core_strike=90, user='lancaster')
    with open('/Users/mike/Desktop/HAAL1a', 'w') as f:
        f.writelines(cif2)
    # cif1 = Cif('/Users/mike/Dropbox/science/harvard/2G_data/mike/MIL/ARM(3000,20)/MIL11').data
    # line0 = cif1.iloc[0].values
    # print(line0[1] == 2)
    # comp = ['ARM', 2, 331.3, -52.9, 331.3, -52.9, 4.13E-06, 000.4, 340.0, 32.0, 0.002359, 0.002268,
    #         0.000730, 'lancaste', '2019-07-18', '15:21:16']
    #
    # for i, v in enumerate(comp):
    #     print(i, v, line0[i], v == line0[i])
    #     # print(i, type(v), type(comp[i]), v==comp[i])
