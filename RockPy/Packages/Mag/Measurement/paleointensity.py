__author__ = 'volk'
import RockPy
from RockPy.core.result import Result

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

from RockPy.Packages.Mag.Measurement.Simulation import paleointensity

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

    @classmethod
    def from_simulation(cls, sobj, idx=0, **simparams):

        method = simparams.pop('method', 'fabian')

        if method == 'fabian':
            simobj = paleointensity.Fabian2001(**simparams)
        return cls(sobj=sobj, mdata=simobj.get_data(), ftype_data=simobj)

    @staticmethod
    def format_jr6(ftype_data, sobj_name=None):
        """

        Returns
        -------

        """

        # check if sample name in file
        if sobj_name not in ftype_data.data['specimen'].values:
            Paleointensity.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Paleointensity.log().error('wrong sample name?')
            return

        # read ftype
        data = ftype_data.data[ftype_data.data['specimen'] == sobj_name].reset_index(drop=True)

        # rename the columns from magic format -> RockPy internal names
        print(data)
        data = data.rename(
            columns={'magn_x': 'x', 'magn_y': 'y', 'magn_z': 'z', 'magn_moment': 'm', 'treat_temp': 'ti'})
        print(data)

        # add tj column:
        # tj := temperature prior to ti step e.g. temperature before ck step
        data['tj'] = np.nan
        data.loc[1:, ('tj')] = data['ti'].values[:-1]

        # delete specimens column
        del data['specimen']
        return data

    @property
    def demag(self):
        """
        Thermal demagnetization steps of the experiments, also giving NRM step

        Returns
        -------
            pandas.DataFrame
        """
        # depending if NRM is added....
        # d = self.data[(self.data['LT_code'] == 'LT-T-Z') | (self.data['LT_code'] == 'LT-NO')].set_index('ti')
        d = self.data[(self.data['LT_code'] == 'LT-T-Z')].set_index('ti')
        return d

    @property
    def acqu(self):
        """
        Acquisition of partial Thermal remanent magnetization steps of the experiments, also giving NRM step.

        Notes
        -----
        This gives the experimental value of the NRM remaining (ti) and pTRM acquisition (ti). The true pTRM gained (ti)
        can be obtained with measurement.ptrm

        Returns
        -------
            pandas.DataFrame
        """
        d = self.data[(self.data['LT_code'] == 'LT-T-I') | (self.data['LT_code'] == 'LT-NO')].set_index('ti')
        return d

    @property
    def ck(self):
        """
        pTRM check steps of the experiments, also giving NRM step

        Returns
        -------
            pandas.DataFrame
        """

        d = self.data[self.data['LT_code'] == 'LT-PTRM-I'].set_index('ti')
        return d

    @property
    def ac(self):
        """
        aditivity check steps of the experiments, also giving NRM step

        Returns
        -------
            pandas.DataFrame
        """

        d = self.data[self.data['LT_code'] == 'LT-PTRM-Z'].set_index('ti')
        return d

    @property
    def tr(self):
        """
        MD tail check steps of the experiments, also giving NRM step

        Returns
        -------
            pandas.DataFrame
        """

        d = self.data[self.data['LT_code'] == 'LT-PTRM-MD'].set_index('ti')
        return d

    @property
    def ptrm(self):
        """
        pTRM acuisition steps of the experiments. Vector substration of the pt[[x,y,z]] and th[[x,y,z]] steps for
        each ti. Also recalculates the moment ['m'] using np.linalg.norm

        Returns
        -------
            pandas.DataFrame
        """

        d = self.acqu.copy()
        d.loc[:, ('x', 'y', 'z')] -= self.demag.loc[:, ('x', 'y', 'z')]
        d['LT_code'] = 'PTRM'
        d['m'] = np.linalg.norm(d[['x', 'y', 'z']], axis=1)

        return d

        ####################################################################################################################

    ####################################################################################################################
    """ RESULTS CALCULATED USING CALCULATE_SLOPE  METHODS """

    class result_slope(Result):

        def recipe_default(self, vmin=20, vmax=700, component='m', **unused_params):
            """
            calculates the least squares slope for the specified temperature interval

            :param parameter:

            """

            # get equal temperature steps for both demagnetization and acquisition measurements
            equal_steps = list(set(self.mobj.demag['level'].values) & set(self.mobj.acqu['level'].values))

            # Filter data for the equal steps and filter steps outside of tmin-tmax range
            # True if step between var_min, var_max
            equal_steps = sorted(i for i in equal_steps if vmin <= i <= vmax)

            # filtering for equal variables
            demag_data = self.mobj.demag.set_index('level').loc[equal_steps]  # filtered data for var_min var_max
            acqu_data = self.mobj.acqu.set_index('level').loc[equal_steps]  # filtered data for var_min var_max

            # data = RockPyData(['demagnetization', 'acquisition'])
            #
            # # setting the data
            # data['demagnetization'] = demag_data[component].v
            # data['acquisition'] = acq_data[component].v
            #
            # try:
            #     slope, sigma, y_int, x_int = data.lin_regress('acquisition', 'demagnetization')
            #     self.results['slope'] = [[[slope, sigma]]]
            #     self.results['sigma'] = sigma
            #     self.results['y_int'] = y_int
            #     self.results['x_int'] = x_int
            #     self.results['n'] = len(demag_data[component].v)
            # except TypeError:
            #     self.log.error('No data found')
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nan



if __name__ == '__main__':
    s = RockPy.Sample('61')
    # m = s.add_measurement(mtype='paleointensity',
    #                       ftype='jr6',
    #                       fpath=os.path.join(RockPy.test_data_path, 'TT-paleointensity.jr6'),
    #                       dialect='tdt')
    #
    # print(m.data)

    m = s.add_simulation(mtype='pint')
    print(m.result_slope())
