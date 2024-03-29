import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from RockPy import ureg
from copy import deepcopy


class Vsm(Ftype):
    standard_calibration_exponent = 0

    mtype_translation = {'Direct moment vs. field; Hysteresis loop\n': ('hys',),
                         'Direct moment vs. field; Initial magnetization; Hysteresis loop\n': ('hys',),
                         'Direct moment vs. field; Multiple segments\n': ('hys',),
                         'Remanence curves:  DCD\n': ('dcd',),
                         'Remanence curves:  IRM + DCD\n': ('irm', 'dcd'),
                         'Direct moment vs. field; First-order reversal curves\n': ('forc',),
                         }

    in_units = {'Field (T)': ureg('tesla'),
                'Remanence (Am2)': ureg('ampere meter ^2')}

    out_units = in_units

    units = {'Field (T)': ureg('tesla'),
             'Remanence (Am2)': ureg('ampere meter ^2')}

    def __init__(self, dfile, snames=None, dialect=None, reload=False):
        """
        Args:
            dfile:
            snames:
            dialect:
            reload:
        """
        self._raw_data = self.read_raw_data(dfile=dfile)

        # get the file infos first -> line numbers needed ect.
        self.file_length = len(self.raw_data)

        mtype, header_end, segment_start, segment_end, segment_widths = self.read_basic_file_info()

        self.mtype = self.mtype_translation[mtype]
        self.header = self.read_header(dfile, header_end)

        super().__init__(dfile, snames=snames, dialect=dialect,
                         reload=reload, header=self.header)

        self.segment_header = self.read_segement_infos(
            dfile, mtype, header_end, segment_start, segment_end, segment_widths)

        # check the calibration factor
        self.calibration_factor = float(self.header.loc['Calibration factor'])

        # self.correct_exp = None
        # if not np.isnan(self.calibration_factor):
        #     if np.floor(np.log10(self.calibration_factor)) != self.standard_calibration_exponent:
        #         self.correct_exp = np.power(10, np.floor(
        #             np.log10(self.calibration_factor)))
        #         RockPy.log.warning(
        #             'CALIBRATION FACTOR (cf) seems to be wrong. CF should be {} here: {}. Data was corrected'.format(
        #                 self.standard_calibration_exponent,
        #                 int(np.floor(np.log10(self.calibration_factor)))))

        # if self.correct_exp:
        #     for c in self.data:
        #         if any(t in c for t in ('(Am2)',)):
        #             self.data[c] *= self.correct_exp

    def read_basic_file_info(self):
        """Opens the file and extracts the mtype, header lines, segment lines,
        segment widths, data lines, and data widths.
        """
        mtype, header_end, segment_start, segment_widths, data_start, data_widths = None, None, None, None, None, None

        empty = []
        for i, l in enumerate(self.raw_data):
            if not l.rstrip():
                empty.append(i)
            if i == 1:
                mtype = l
            if 'Number of data' in l:
                header_end = i
            if '0000001' in l:
                segment_widths = [len(n) + 1 for n in l.split(',')]
                segment_start = i
            if l.startswith('+') or l.startswith('-'):
                self.data_start = i
                self.data_widths = [len(n) for n in l.split(',')]
                break

        segment_end = max(i for i in empty if i < self.data_start)
        return mtype, header_end, segment_start, segment_end, segment_widths

    def read_header(self, dfile, header_end):
        """Function reads the header file

        Args:
            dfile:
            header_end:
        """
        head = self.raw_data[:header_end]
        header = pd.read_fwf(io.StringIO(''.join(head)),
                             skiprows=2, skip_blank_lines=True,
                             widths=(31, 13), index_col=0, names=[0])
        # remove empty line and section headers
        idx = [i for i, v in enumerate(header.index) if str(
            v).upper() != v if str(v) != 'nan']

        header = header.iloc[idx]
        header = header.replace('No', False)
        header = header.replace('Yes', True)
        # make numerical entries to float
        for i, v in enumerate(header.index):
            try:
                header.iloc[i] = float(header.iloc[i])
            except:
                pass

        # add file location to header
        header.loc['fpath'] = dfile

        if 'Calibration factor' not in header.index:
            header.loc['Calibration factor'] = None
        return header

    def read_segement_infos(self, dfile, mtype,
                            header_end, segment_start, segment_end, segment_widths,
                            ):
        """reads the segments of the VSM file

        Args:
            dfile:
            mtype:
            header_end:
            segment_start:
            segment_end:
            segment_widths:

        Notes:
            VSM - FORC measurements do not have a segments part -> returns None

            Returns -------s
        """

        if 'First-order reversal curves' not in mtype:
            # reading segments_tab data
            head = self.raw_data[header_end + 1:segment_start]

            head = pd.read_fwf(io.StringIO(''.join(head)), names=[],
                               widths=segment_widths)
            segment_header = [' '.join([str(n) for n in line]).replace(
                'nan', '').strip() for line in head.values.T]

            segment_infos = pd.read_csv(dfile, skiprows=segment_start, nrows=int(self.header[0]['Number of segments']),
                                        names=segment_header, encoding='latin-1',
                                        )
            # add column with start indices for each segment
            segment_infos['Start Index'] = [0] + \
                list(segment_infos['Final Index'].values[:-1] + 1)
            # add one to the final index because of empty row
            segment_infos['Final Index'] = [v + i for i,
                                            v in enumerate(segment_infos['Final Index'])]

        else:
            # constructs segment header from file itself
            segment_infos = self._construct_segment_infos_from_data()

        return segment_infos

    def _construct_segment_infos_from_data(self):
        """Uses the Nan rows in the data to construct the segment_info
        DataFrame.

        Returns:
            pd.DataFrame: with segment infos. columns : 'Segment Number',
            'Averaging Time', 'Initial Field', 'Field Increment','Final Field',
            'Pause', 'Final Index'
        """
        segment_infos = pd.DataFrame(columns=['Segment Number', 'Averaging Time', 'Initial Field', 'Field Increment',
                                              'Final Field', 'Pause', 'Final Index'])

        # an empty row == only nan values separates the different segments
        # get nan indices
        nanidx = np.where(~self.data.iloc[:, 1].notnull())[0]

        Binit = []
        step = []
        Bfin = []

        for i, idx in enumerate(nanidx):
            # get start index of segment
            sidx = 0 if i == 0 else nanidx[i - 1] + 1
            # end index
            eidx = idx - 1

            Binit.append(self.data['Field (T)'].iloc[sidx])
            Bfin.append(self.data['Field (T)'].iloc[eidx])

            if Binit[i] != Bfin[i]:
                step.append(
                    np.mean(np.diff(self.data['Field (T)'].iloc[sidx:eidx + 1])))
            else:
                step.append(np.nan)

        segment_infos['Segment Number'] = np.arange(len(nanidx))
        segment_infos['Initial Field'] = Binit
        segment_infos['Field Increment'] = step
        segment_infos['Final Field'] = Bfin
        segment_infos['Final Index'] = nanidx
        segment_infos['Pause'] = self.header.loc['Averaging time'].iloc[0, 0]
        segment_infos['Averaging Time'] = self.header.loc['Averaging time'].iloc[0, 0]

        # add column with start indices for each segment
        segment_infos['Start Index'] = [0] + \
            list(segment_infos['Final Index'].values[:-1] + 1)
        return segment_infos

    def read_file(self):
        # reading data
        data_header = [' '.join([str(n) for n in line]).replace('nan', '').replace('�', '2').replace('²', '2').strip() for line in
                       pd.read_fwf(self.dfile, skiprows=self.data_start - 5, columns=[],
                                   nrows=2, widths=self.data_widths,
                                   encoding='latin1',).values.T]

        return pd.read_csv(io.StringIO(''.join(self.raw_data[self.data_start:-2])),
                           names=data_header,
                           skip_blank_lines=False, squeeze=True,
                           encoding='latin1',
                           )

    @property
    def iter_segments(self):
        """Generator that cycles through the segments :returns: :rtype:
        pandas.DataFrame
        """

        # iterate over individual rows (segments) in the header
        for i, seg in self.segment_header.iterrows():
            segment = self.data.iloc[int(seg['Start Index']):int(
                seg['Final Index'])].dropna(axis=0)
            yield segment

    @property
    def segments(self):
        """returns a list of individual pandas DataFrames for segments :returns:
        :rtype: list
        """
        return list(self.iter_segments)

    def get_segment_data(self, segment_index):
        """Returns the segment of a measurement corresponding to the index
        (segment_index).

        This is used in read file functions such as DCD, if more than one
        measurement is stored in a single run, e.g. (IRM,DCD).

        Args:
            segment_index (int): the index of the segment

        Returns:
            pandas Dataframe:
        """
        return list(self.iter_segments)[segment_index]


if __name__ == '__main__':
    # dcd = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/dcd_vsm.001')
    # hys = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/VSM/hys_vsm.001')
    # print(Vsm(dfile='/Users/mike/Dropbox/science/_projects/RockPy/RockPy/tests/test_data/VSM/dcd_vsm.001').header)

    # s = RockPy.Sample('test')
    # m = s.add_measurement(
    #     fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/VSM/dcd_irm_vsm.001')
    # print(m.data)

    powder = Vsm(
        '/Users/mike/Dropbox/science/harvard/2021_MUC_VSM/doped_Verezosca_750nmFe3O4_120mg.hys')
