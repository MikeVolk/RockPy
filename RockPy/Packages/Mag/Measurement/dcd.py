__author__ = 'volk'
import RockPy
import logging
import numpy as np
import os
import scipy as sp
from scipy import stats
from scipy.stats import lognorm
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from RockPy.core.measurement import Measurement
from RockPy.core.result import Result
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

class Dcd(Measurement):
    logger = logging.getLogger('RockPy.MEASUREMENT.Backfield')
    """
    A Backfield Curve can give information on:
       Bcr: the remanence coercivity

       S300: :math:`(1 - (M_{300mT} /M_{rs})) / 2`

    Bcr is typically determined by finding the intersection of the linear interpolated measurement data with the axis
    representing zero-magnetization.
    For the calculation of S300, the initial magnetization is used as an approximation of the saturation remanence
    :math:`M_{rs}` and the magnetization at 300mT :math:`M_{300mT}` is determined by linear interpolation of measured
    data.

    Possible data structure::

       self.data: the remanence measurement after the field was applied (normal measurement mode for e.g. VFTB or VSM)


    """
    # _visuals = (('backfield',
    #              {'features':('zero_lines', 'backfield_data', 'rockmag_results'), 'color':'k', 'marker':'.'}),
    #             )
    @property
    def data(self):
        out = super().data
        return out.set_index('B')
    ####################################################################################################################

    @classmethod
    def from_simulation(cls, sobj, idx=None,
                        ms=250., bmax=0.5, E=0.005, G=0.3, steps=20, log_steps=False,
                        noise=None, color=None, marker=None, linestyle=None): #todo implement simulation
        """
        Simulates a backfield measurement based on a single log-normal gaussian distribution.

        E:  Median destructive field - represents the mean value of the log-Gaussian distribution, and therefore, the
            logarithmic field value of the maximum gradient.

        G:  G describes the standard deviation or half-width of the distribution.

        """
        raise NotImplementedError

    ####################################################################################################################
    """ formatting functions """

    @staticmethod
    def format_vsm(ftype_data, sobj_name=None):
        '''
        formatting routine for VSM type hysteresis measurements. Indirectly called by the measurement.from_file method.

        Parameters
        ----------
        ftype_data: RockPy.io
            data read by Mag.io.Vsm.vsm function.
            Contains:
              - data
              - header
              - segment_header

        sobj_name: str
            unused

        Returns
        -------
            data: pandas.Dataframe
                columns = ['B','M']
            ftype_data: io object
                io object as read by Mag.io.Vsm.vsm


        '''

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['Field (T)', 'Remanence (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            print(ftype_data.data.columns)
            Dcd.log().error('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        data = ftype_data.data.rename(columns={"Field (T)": "B", "Moment (Am2)": "M"})
        return data

    @staticmethod
    def format_vftb(ftype_data, sobj_name=None): #todo implement VFTB
        '''
        formats the output from vftb to measurement.data
        :return:
        '''
        raise NotImplementedError

    ####################################################################################################################
    ''' Mrs '''

    class mrs(Result):
        def recipe_linear(self, **non_method_parameters):
            """
            Magnetic Moment at first measurement point
            :param parameter:
            :return:
            """
            raise NotImplementedError


    ####################################################################################################################
    ''' Bcr '''

    class bcr(Result):
        '''
        Calculates the coercivity of remanence from the dcd curve
        '''
        default_recipe = 'linear'


        def recipe_linear(self, no_points=4, check=False, **non_method_parameters):
            """
            Calculates the coercivity using a linear interpolation between the points crossing the x axis for upfield and down field slope.

            Parameters
            ----------
                field_limit: float
                    default: 0, 0mT
                    the maximum/ minimum fields used for the linear regression

            Note
            ----
                Uses scipy.linregress for calculation
            """
            # initialize result
            result = []

            # get magneization limits for a calculation using the 2 points closest to 0 fro each direction
            moment = sorted(abs(self.data['data']['mag'].v))[no_points - 1]

            # filter data for fields higher than field_limit
            data = self.data['data'].filter(abs(self.data['data']['mag'].v) <= moment)

            # calculate bcr
            slope, intercept, r_value, p_value, std_err = stats.linregress(data['field'].v, data['mag'].v)
            result.append(abs(intercept / slope))
            # check plot
            if check:
                x = data['field'].v
                y_new = slope * x + intercept
                plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy3.colorscheme[0])
                plt.plot(x, y_new, color=RockPy3.colorscheme[0])

            # check plot
            if check:
                plt.plot([-np.nanmean(result)], [0], 'xk')
                plt.grid()
                plt.show()

            self.results['bcr'] = [[(np.nanmean(result), np.nan)]]

    def recipe_nonlinear(self, no_points=4, check=False, **non_method_parameters):
        """
        Calculates the coercivity of remanence using a spline interpolation between the points crossing
        the x axis for upfield and down field slope.

        Parameters
        ----------
            field_limit: float
                default: 0, 0mT
                the maximum/ minimum fields used for the linear regression

        Note
        ----
            Uses scipy.linregress for calculation
        """
        # initialize result
        result = []

        # get limits for a calculation using the no_points points closest to 0 fro each direction
        limit = sorted(abs(self.data['data']['mag'].v))[no_points - 1]
        # the field_limit has to be set higher than the lowest field
        # if not the field_limit will be chosen to be 2 points for uf and df separately
        if no_points < 2:
            self.logger.warning('NO_POINTS INCOMPATIBLE minimum 2 required' % (no_points))
            self.logger.warning('\t\t setting NO_POINTS - << 2 >> ')
            self.calculation_parameter['bcr']['no_points'] = 2

        # filter data for fields higher than field_limit
        data = self.data['data'].filter(abs(self.data['data']['mag'].v) <= limit)  # .sort('field')
        x = np.linspace(data['field'].v[0], data['field'].v[-1])

        spl = UnivariateSpline(data['field'].v, data['mag'].v)
        y_new = spl(x)
        idx = np.argmin(abs(y_new))
        result = abs(x[idx])

        if check:
            plt.plot(data['field'].v, data['mag'].v, '.', color=RockPy3.colorscheme[0])
            plt.plot(x, y_new, color=RockPy3.colorscheme[0])
            plt.plot(-result, 0, 'xk')
            plt.grid()
            plt.show()

        # set result so it can be accessed
        self.results['bcr'] = [[(np.nanmean(result), np.nanstd(result))]]


if __name__ == '__main__':
    S = RockPy.Study()
    s = S.add_sample(sname='test')
    m = s.add_measurement(fpath='/Users/mike/Dropbox/github/2016-FeNiX.2/data/(HYS,DCD)/FeNiX_FeNi00-Fa36-G01_(IRM,DCD)_VSM#36.5mg#(ni,0,perc)_(gc,1,No).001',
                            mtype='dcd',
                            ftype='vsm')
    print(m.data)
