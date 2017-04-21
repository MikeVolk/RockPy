import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy


class Vsm(Ftype):
    def __init__(self, dfile, snames=None, dialect=None):
        super().__init__(dfile, snames=snames, dialect=dialect)

        with open(self.dfile, 'r', encoding="ascii", errors="surrogateescape") as f:
            for i, l in enumerate(f.readlines()):
                if i == 1:
                    mtype = l
                if 'Number of data' in l:
                    header_end = i
                if '0000001' in l:
                    widths = [len(n)+1 for n in l.split(',')]
                    segment_start = i
                if l.startswith('+') or l.startswith('-'):
                    data_start = i
                    data_widths = [len(n) for n in l.split(',')]
                    break

        self.mtype = mtype

        # reading the header
        header = pd.read_fwf(self.dfile,
                             skiprows=2, nrows=header_end - 1, skip_blank_lines=True,
                             widths=(31, 13), index_col=0, names=[0]).dropna()
        header = header.replace('No', False)
        header = header.replace('Yes', True)

        self.header = header

        if not 'First-order reversal curves' in mtype:
            # reading segments_tab data
            segment_header = [' '.join([str(n) for n in line]).replace('nan', '').strip() for line in
                              pd.read_fwf(self.dfile, skiprows=header_end + 1, nrows=segment_start - header_end -2,
                                          widths=widths, header=None).values.T]
            segments = pd.read_csv(self.dfile, skiprows=segment_start, nrows=int(header[0]['Number of segments']),
                                   names=segment_header,
                                   )

            self.segments_tab = segments

        # reading data
        data_header = [' '.join([str(n) for n in line]).replace('nan', '').replace('�', '2').strip() for line in
                       pd.read_fwf(self.dfile, skiprows=data_start - 4,
                                   nrows=3, widths=data_widths).values.T]

        data = pd.read_csv(self.dfile, skiprows=data_start,
                           nrows=int(header[0]['Number of data'])+int(header[0]['Number of segments'])-1,
                           names=data_header, skip_blank_lines=False, squeeze=True,
                           )
        self.data = data#.dropna(axis=0)

    @property
    def segments(self):
        """
        Generator that cycles through the segments
        Returns
        -------
            pandas.DataFrame
        """
        indices = [0] + [seg['Final Index']+i for i, seg in self.segments_tab.iterrows()]
        for i, idx in enumerate(indices[:-1]):
            yield self.data.loc[indices[i]:indices[i+1]].dropna(axis=0)

if __name__ == '__main__':
    dcd = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/dcd_vsm.001')
    hys = Vsm(dfile='/Users/mike/github/RockPy/RockPy/tests/test_data/hys_vsm.001')
