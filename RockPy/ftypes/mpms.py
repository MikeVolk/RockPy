import pandas as pd
import numpy as np
import io
from copy import deepcopy

import RockPy

class Mpms(RockPy.core.ftype.Ftype):
    def __init__(self, dfile, snames=None, dialect=None, reload=False):
        """
        Args:
            dfile:
            snames:
            dialect:
            reload:
        """
        super().__init__(dfile, snames=snames, dialect=dialect, reload = reload)

    def get_data_index(self):
        """Opens the file and searches for the `[Data]` line , returns the index
        f that line.
        """

        with open(self.dfile) as f:
            for i, l in enumerate(f.readlines()):
                if '[Data]' in l:
                    return i+1

    def read_file(self):
        # reading data
        data_start_index = self.get_data_index()

        data = pd.read_csv(self.dfile, skiprows=data_start_index, comment='#',
                           squeeze=True)

        self._get_comment_line_indices()
        return data

    def group_by(self, what):
        """iterator that returns a pandas.DataFrame for each entry of `what`

        Examples:
            This is usefull to quickly looking at field or amplitude dependent
            susceptibilities

        Args:
            what (to be grouped by):

        Returns:
            iterator - pandas.DataFrame:
        """

        if not what in self.data.columns:
            raise KeyError('Value << %s >> not in data.columns. Chose from: %s'%(what, ', '.join(self.data.columns)))

        for v in sorted(set(self.data[what])):
            yield self.data[self.data[what] == v]

if __name__ == '__main__':
    d = Mpms('/Users/mike/Dropbox/github/2017-monoclinic_pyrrhotite/data/MPMS/c-axis/LTPY_MSM17591-1-1_MPMS_(ACsus 7f)-mg.BparallelC.005.ac.dat')

    print(d.group_by('Wave Frequency (Hz)'))