import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy


class Vsm(Ftype):
    standard_calibration_exponent = 0

    mtype_translation = {'Direct moment vs. field; Hysteresis loop\n': ('hys',),
                         'Direct moment vs. field; Initial magnetization; Hysteresis loop\n': ('hys',),
                         'Remanence curves:  DCD\n': ('dcd',),
                         'Remanence curves:  IRM + DCD\n':('irm','dcd'),
                         }

    def __init__(self, dfile, snames=None, dialect=None, reload=False):

        # get the file infos first -> line numbers needed ect.
        mtype, header_end, segment_start, segment_widths, self.data_start, self.data_widths, self.file_length = self.read_basic_file_info(
            dfile)

        self.mtype = self.mtype_translation[mtype]
        self.header = self.read_header(dfile, header_end)
        self.segment_header = self.read_segement_infos(dfile, mtype, header_end, segment_start, segment_widths)

        super().__init__(dfile, snames=snames, dialect=dialect, reload = reload)

        # check the calibration factor
        self.calibration_factor = float(self.header.T['Calibration factor'])

        if np.floor(np.log10(self.calibration_factor)) != self.standard_calibration_exponent:
            self.correct_exp = np.power(10, np.floor(np.log10(self.calibration_factor)))
            RockPy.log.warning(
                'CALIBRATION FACTOR (cf) seems to be wrong. CF should be {} here: {}. Data was corrected'.format(
                    self.standard_calibration_exponent,
                    int(np.floor(np.log10(self.calibration_factor)))))
        else:
            self.correct_exp = None

        if self.correct_exp:
            for c in self.data:
                if any(t in c for t in ('(Am2)',)):
                    self.data[c] *= self.correct_exp

    def read_basic_file_info(self, dfile):
        '''
        Opens the file and extracts the mtype, header lines, segment lines, segment widths, data lines, and data widths.
        '''
        mtype, header_end, segment_start, segment_widths, data_start, data_widths = None, None, None, None, None, None

        with open(dfile, 'r', encoding="ascii", errors="surrogateescape") as f:
            d = f.readlines()
            file_length = len(d)
            for i, l in enumerate(d):
                if i == 1:
                    mtype = l
                if 'Number of data' in l:
                    header_end = i
                if '0000001' in l:
                    segment_widths = [len(n) + 1 for n in l.split(',')]
                    segment_start = i
                if l.startswith('+') or l.startswith('-'):
                    data_start = i
                    data_widths = [len(n) for n in l.split(',')]
                    break

        return mtype, header_end, segment_start, segment_widths, data_start, data_widths, file_length

    def read_header(self, dfile, header_end):
        '''
        Function reads the header file

        Returns
        -------

        '''
        header = pd.read_fwf(dfile,
                             skiprows=2, nrows=header_end - 1, skip_blank_lines=True,
                             widths=(31, 13), index_col=0, names=[0])

        # remove empty line and section headers
        idx = [i for i,v in enumerate(header.index) if not str(v).upper() == v if str(v) != 'nan']

        header = header.iloc[idx]
        header = header.replace('No', False)
        header = header.replace('Yes', True)
        header.loc['fpath'] = dfile
        return header

    def read_segement_infos(self, dfile,
                            mtype, header_end, segment_start, segment_widths,
                            ):
        '''
        reads the segments of the VSM file

        Notes
        -----
        VSM - FORC measurements do not have a segments part -> returns None

        Returns
        -------

        '''

        if not 'First-order reversal curves' in mtype:
            # reading segments_tab data
            segment_header = [' '.join([str(n) for n in line]).replace('nan', '').strip() for line in
                              pd.read_fwf(dfile, skiprows=header_end + 1, nrows=segment_start - header_end - 2,
                                          widths=segment_widths, header=None).values.T]
            segment_infos = pd.read_csv(dfile, skiprows=segment_start, nrows=int(self.header[0]['Number of segments']),
                                        names=segment_header, encoding='latin-1',
                                        )
        else:
            # constructs segment header from file itself
            segment_infos = self._construct_segment_infos_from_data()

        # add column with start indices for each segment
        segment_infos['Start Index'] = [0] + list(segment_infos['Final Index'].values[:-1] + 1)

        return segment_infos

    def _construct_segment_infos_from_data(self):
        """
        Uses the Nan rows in the data to construct the segment_info DataFrame.

        Returns
        -------
            pd.DataFrame
                with segment infos.
                columns : 'Segment Number', 'Averaging Time', 'Initial Field', 'Field Increment','Final Field', 'Pause', 'Final Index'
        """
        segment_header = pd.DataFrame(columns=['Segment Number', 'Averaging Time', 'Initial Field', 'Field Increment',
                                               'Final Field', 'Pause', 'Final Index'])

        # an empty row == only nan values separates the different segments
        # get nan indices
        nanidx = np.where(~self.data.iloc[:, 1].notnull())[0]

        Binit = []
        step = []
        Bfin = []

        for i, idx in enumerate(nanidx):
            # get start index of segment
            if i == 0:
                sidx = 0
            else:
                sidx = nanidx[i - 1] + 1
            # end index
            eidx = idx - 1

            Binit.append(self.data['Field (T)'].iloc[sidx])
            Bfin.append(self.data['Field (T)'].iloc[eidx])

            if Binit[i] != Bfin[i]:
                step.append(np.mean(np.diff(self.data['Field (T)'].iloc[sidx:eidx + 1])))
            else:
                step.append(np.nan)

        segment_header['Segment Number'] = np.arange(len(nanidx))
        segment_header['Initial Field'] = Binit
        segment_header['Field Increment'] = step
        segment_header['Final Field'] = Bfin
        segment_header['Final Index'] = nanidx
        segment_header['Pause'] = self.header.loc['Averaging time'].iloc[0, 0]
        segment_header['Averaging Time'] = self.header.loc['Averaging time'].iloc[0, 0]

        return segment_header

    def read_file(self):

        # reading data
        data_header = [' '.join([str(n) for n in line]).replace('nan', '').replace('�', '2').strip() for line in
                       pd.read_fwf(self.dfile, skiprows=self.data_start - 4,
                                   nrows=3, widths=self.data_widths).values.T]

        data = pd.read_csv(self.dfile, skiprows=self.data_start,
                           nrows=int(self.file_length - self.data_start) - 2,
                           names=data_header, skip_blank_lines=False, squeeze=True,
                           )
        return data

    @property
    def iter_segments(self):
        """
        Generator that cycles through the segments
        Returns
        -------
            pandas.DataFrame
        """

        # iterate over individual rows (segments) in the header
        for i, seg in self.segment_header.iterrows():
            segment = self.data.iloc[int(seg['Start Index']):int(seg['Final Index'])].dropna(axis=0)
            yield segment

    @property
    def segments(self):
        """
        returns a list of individual pandas DataFrames for segments
        Returns
        -------
            list
        """
        return list(self.iter_segments)


    def get_segment_data(self, segment_index):
        """
        Returns the segment of a measurement corresponding to the index (segment_index).

        This is used in read file functions such as DCD, if more than one measurement is stored in a single run,
        e.g. (IRM,DCD).

        Parameters
        ----------
        segment_index: int
            the index of the segment

        Returns
        -------
            pandas Dataframe
        """
        return list(self.iter_segments)[segment_index]

if __name__ == '__main__':
    # dcd = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/dcd_vsm.001')
    # hys = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/VSM/hys_vsm.001')
    print(Vsm(dfile='/Users/mike/Dropbox/github/collaborations/Andrew Fowler (UMN)/data/VSM/pyrrhotite#3base_7.2mg_DCD_VSM.001').header)
