__author__ = 'volk'
import RockPy

from copy import deepcopy
from math import tanh, cosh

import numpy as np
import numpy.random
import scipy as sp
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from lmfit import minimize, Parameters, report_fit
import matplotlib.dates
import datetime

from RockPy.core import measurement


class Paleointensity(measurement.Measurement):

    # def __init__(self, sobj,
    #              fpath=None, ftype=None, mdata=None,
    #              ftype_data = None,
    #              series=None,
    #              idx=None,
    #              **options
    #              ):
    #     super().__init__(fpath=fpath, ftype=ftype,
    #                      ftype_data=ftype_data,
    #                      mdata=mdata,
    #                      series=series,
    #                      idx=idx,
    #                      **options
    #                      )
    @staticmethod
    def format_jr6(ftype_data, sobj_name=None):
        """

        Returns
        -------

        """
        if sobj_name not in ftype_data.data['specimen'].values:
            Paleointensity.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Paleointensity.log().error('wrong sample name?')
            return

        data = ftype_data.data[ftype_data.data['specimen']==sobj_name].reset_index(drop=True)
        data['tj'] = np.nan
        data['tj'].iloc[1:] = data['treat_temp'].values[:-1]
        del data['specimen']
        return data

    @property
    def th(self):
        d = self.data[(self.data['LT_code'] == 'LT-T-Z') | (self.data['LT_code'] == 'LT-NO')].set_index('treat_temp')
        return d

    @property
    def pt(self):
        d = self.data[(self.data['LT_code'] == 'LT-T-I') | (self.data['LT_code'] == 'LT-NO')].set_index('treat_temp')
        return d\

    @property
    def ck(self):
        d = self.data[self.data['LT_code'] == 'LT-PTRM-I'].set_index('treat_temp')
        return d

    @property
    def ptrm(self):
        d = deepcopy(self.pt)
        d[['x','y','z']] = self.pt[['x','y','z']] - self.th[['x','y','z']]
        d['LT_code'] = 'PTRM'
        d['m'] = np.linalg.norm(d[['x','y','z']], axis=1)

        return d

    ####################################################################################################################

if __name__ == '__main__':
    s = RockPy.Sample('61')
    m = s.add_measurement(mtype='paleointensity',
                          ftype='jr6',
                          fpath=os.path.join(RockPy.test_data_path, 'TT-paleointensity.jr6'),
                          dialect='tdt')
    print(m.ptrm)