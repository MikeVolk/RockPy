import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy


class Vsm(Ftype):
    standard_calibration_exponent = 0

    def __init__(self, dfile, snames=None, dialect=None):

        # get the file infos first -> line numbers needed ect.
        mtype, header_end, segment_start, segment_widths, self.data_start, self.data_widths, self.file_length = self.read_basic_file_info(dfile)

        self.header = self.read_header(dfile, header_end)
        self.segment_header = self.read_segement_infos(dfile, mtype, header_end, segment_start, segment_widths)

        super().__init__(dfile, snames=snames, dialect=dialect)

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
                if any(t in c for t in ('(Am2)', )):
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
                    segment_widths = [len(n)+1 for n in l.split(',')]
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
                             widths=(31, 13), index_col=0, names=[0]).dropna()
        header = header.replace('No', False)
        header = header.replace('Yes', True)

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
        segment_infos = None

        if not 'First-order reversal curves' in mtype:
            # reading segments_tab data
            segment_header = [' '.join([str(n) for n in line]).replace('nan', '').strip() for line in
                              pd.read_fwf(dfile, skiprows=header_end + 1, nrows=segment_start - header_end -2,
                                          widths=segment_widths, header=None).values.T]
            segment_infos = pd.read_csv(dfile, skiprows=segment_start, nrows=int(self.header[0]['Number of segments']),
                                   names=segment_header, encoding = 'latin-1',
                                   )
        return segment_infos

    def read_file(self):

        # reading data
        data_header = [' '.join([str(n) for n in line]).replace('nan', '').replace('�', '2').strip() for line in
                       pd.read_fwf(self.dfile, skiprows=self.data_start - 4,
                                   nrows=3, widths=self.data_widths).values.T]

        data = pd.read_csv(self.dfile, skiprows=self.data_start,
                           nrows=int(self.file_length-self.data_start)-2,
                           names=data_header, skip_blank_lines=False, squeeze=True,
                           )
        return data


    @property
    def segments(self):
        """
        Generator that cycles through the segments
        Returns
        -------
            pandas.DataFrame
        """
        # indices of the first row of each segment
        indices = [0] + [seg['Final Index']+i for i, seg in self.segment_header.iterrows()]
        for i, idx in enumerate(indices[:-1]):
            yield self.data.loc[indices[i]:indices[i+1]].dropna(axis=0)
    @property
    def segment_list(self):
        return list(self.segments)

    @property
    def segment_list(self):
        return list(self.segments)
    
if __name__ == '__main__':
    # dcd = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/dcd_vsm.001')
    # hys = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/VSM/hys_vsm.001')
    print(Vsm(dfile='/Users/mike/Google Drive/LHMGt4000-12_VSM_preisach_try.001').data)
