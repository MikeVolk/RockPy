__author__ = 'volk'
import RockPy
from RockPy.core.result import Result
from RockPy.core.utils import lin_regress

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

    def equal_acqu_demag_steps(self, vmin=20, vmax=700):
        """
        Filters the th and ptrm data so that the temperatures are within vmin, vmax and only temperatures in both
        th and ptrm are returned.
        """

        # get equal temperature steps for both demagnetization and acquisition measurements
        equal_steps = self.get_values_in_both(self.zf_steps, self.if_steps, 'level')

        # Filter data for the equal steps and filter steps outside of tmin-tmax range
        # True if step between vmin, vmax
        equal_steps = sorted(i for i in equal_steps if vmin <= i <= vmax)

        # filtering for equal variables
        y = self.zf_steps.set_index('level').loc[equal_steps]  # filtered data for vmin vmax
        x = self.ifzf_diff.set_index('level').loc[equal_steps]  # filtered data for vmin vmax

        return x, y

    @classmethod
    def from_simulation(cls, sobj, idx=0, series=None, **simparams):

        pressure_demag = simparams.pop('pressure_demag', False)
        method = simparams.pop('method', 'fabian')

        if method == 'fabian':
            simobj = paleointensity.Fabian2001(**simparams)
        return cls(sobj=sobj, mdata=simobj.get_data(pressure_demag=pressure_demag),
                   series = series,
                   ftype_data=simobj, ftype='simulation')

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
        data = data.rename(
            columns={'magn_x': 'x', 'magn_y': 'y', 'magn_z': 'z', 'magn_moment': 'm', 'treat_temp': 'ti'})

        # add tj column:
        # tj := temperature prior to ti step e.g. temperature before ck step
        data['tj'] = np.nan
        data.loc[1:, ('tj')] = data['ti'].values[:-1]

        # delete specimens column
        del data['specimen']
        return data

    @staticmethod
    def format_tdt(ftype_data, sobj_name=None):
        """

        Returns
        -------

        """

        # check if sample name in file
        if not ftype_data.has_specimen(sobj_name):
            return


        # read ftype
        data = ftype_data.data[ftype_data.data['specimen'] == sobj_name].reset_index(drop=True)

        # rename the columns from magic format -> RockPy internal names
        data = data.rename(
            columns={'magn_x': 'x', 'magn_y': 'y', 'magn_z': 'z', 'magn_moment': 'm', 'step': 'ti'})

        # # delete specimens column
        del data['specimen']
        del data['dec']
        del data['inc']
        return data

    @property
    def zf_steps(self):
        """
        Thermal demagnetization steps of the experiments, also giving NRM step

        Returns
        -------
            pandas.DataFrame
        """
        d = self.data[(self.data['LT_code'] == 'LT-T-Z') | (self.data['LT_code'] == 'LT-NO')].set_index('ti')
        return d

    @property
    def if_steps(self):
        """
        Acquisition of partial Thermal remanent magnetization steps of the experiments, also giving NRM step.

        Notes
        -----
        This gives the experimental value of the NRM remaining (ti) and pTRM acquisition (ti). The true pTRM gained (ti)
        can be obtained with measurement.ifzf_diff

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
        additivity check steps of the experiments, also giving NRM step

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
    def ifzf_diff(self):
        """
        pTRM acuisition steps of the experiments. Vector substration of the pt[[x,y,z]] and th[[x,y,z]] steps for
        each ti. Also recalculates the moment ['m'] using np.linalg.norm

        Returns
        -------
            pandas.DataFrame
        """

        equal_vals = self.get_values_in_both(self.if_steps, self.zf_steps, 'level')


        if_steps = self.if_steps[np.in1d(self.if_steps['level'], equal_vals)].copy()
        zf_steps = self.zf_steps[np.in1d(self.zf_steps['level'], equal_vals)].copy()

        # print(if_steps)
        # print(zf_steps)
        if_steps.loc[:, ('x', 'y', 'z')] -= zf_steps.loc[:, ('x', 'y', 'z')]
        if_steps['LT_code'] = 'PTRM'
        if_steps['m'] = np.sqrt(if_steps.loc[:, ['x', 'y', 'z']].apply(lambda x: x ** 2).sum(axis=1))
        return if_steps

    ####################################################################################################################

    ####################################################################################################################
    """ RESULTS CALCULATED USING CALCULATE_SLOPE  METHODS """

    class slope(Result):
        # __calculates__ = ['sigma', 'yint', 'xint', 'n']

        def vd(self, vmin, vmax,
               **unused_params):
            """
            Vector differences

            :param parameter:
            :return:

            """
            acqu, demag = self.mobj.equal_acqu_demag_steps(vmin=vmin, vmax=vmax)
            vd = np.diff(demag.loc[:, ['x', 'y', 'z']], axis=0)
            return vd.astype(float)

        def vds(self, vmin, vmax, **unused_params): #todo move somwhere else
            """
            The vector difference sum of the entire NRM vector :math:`\\mathbf{NRM}`.

            .. math::

               VDS=\\left|\\mathbf{NRM}_{n_{max}}\\right|+\\sum\\limits_{i=1}^{n_{max}-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}

            where :math:`\\left|\\mathbf{NRM}_{i}\\right|` denotes the length of the NRM vector at the :math:`i^{demagnetization}` step.

            Parameters
            ----------
                vmin: float
                vmax: float
                recalc: bool
                non_method_parameters: dict
            """
            acqu, demag = self.mobj.equal_acqu_demag_steps(vmin=vmin, vmax=vmax)
            NRM_var_max = np.linalg.norm(self.mobj.zf_steps[['x', 'y', 'z']].iloc[-1])
            NRM_sum = np.sum(np.linalg.norm(self.vd(vmin=0, vmax=700), axis=1))
            return abs(NRM_var_max) + NRM_sum

        def x_dash(self, vmin, vmax, component,
                   **unused_params):
            """

            :math:`x_0 and :math:`y_0` the x and y points on the Arai plot projected on to the best-fit line. These are
            used to
            calculate the NRM fraction and the length of the best-fit line among other parameters. There are
            multiple ways of calculating :math:`x_0 and :math:`y_0`, below is one example.

            ..math:

              x_i' = \\frac{1}{2} \\left( x_i + \\frac{y_i - Y_{int}}{b}


            :param parameter:
            :return:

            """

            demagnetization, acquisition = self.filter_demagnetization_ptrm(vmin=vmin, vmax=vmax)
            x_dash = (
                demagnetization[component].v - self.result_y_int(vmin=vmin, vmax=vmax, component=component)[0])
            x_dash = x_dash / self.result_slope(vmin=vmin, vmax=vmax, component=component)[0]
            x_dash = acquisition[component].v + x_dash
            x_dash = x_dash / 2.

            return x_dash

        def y_dash(self, vmin, vmax, component,
                   **unused_params):
            """

            :math:`x_0` and :math:`y_0` the x and y points on the Arai plot projected on to the best-fit line. These are
            used to
            calculate the NRM fraction and the length of the best-fit line among other parameters. There are
            multiple ways of calculating :math:`x_0` and :math:`y_0`, below is one example.

            ..math:

               y_i' = \\frac{1}{2} \\left( y_i + bx + Y_{int} \\right)


            Notes
            -----
                needs slope and yint. Classes that use this directly or indirectly need 
                dependencies = ('slope', 'yint')

            """
            acqu_data, demag_data = self.mobj.equal_acqu_demag_steps(vmin=vmin, vmax=vmax)

            y_dash = acqu_data[component] + (self.get_result('slope') * demag_data[component])\
                            + self.get_result('yint')

            return 0.5 * y_dash.values

        def delta_x_dash(self, vmin, vmax, component,
                         **unused_params):
            """

            :math:`\Delta x_0` is the TRM length of the best-fit line on the Arai plot.

            """
            x_dash = self.x_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            out = abs(np.max(x_dash) - np.min(x_dash))
            return out

        def delta_y_dash(self, vmin, vmax, component,
                         **unused_params):
            """

            :math:`\Delta y_0`  is the NRM length of the best-fit line on the Arai plot.

            """
            y_dash = self.y_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            out = abs(np.max(y_dash) - np.min(y_dash))
            return out

        def best_fit_line_length(self, vmin=20, vmax=700, component='m'):
            L = np.sqrt((self.delta_x_dash(vmin=vmin, vmax=vmax, component=component)) ** 2 +
                        (self.delta_y_dash(vmin=vmin, vmax=vmax, component=component)) ** 2)
            return L

        def recipe_default(self, vmin=20, vmax=700, component='m', **unused_params):
            """
            calculates the least squares slope for the specified temperature interval

            Parameters
            ----------
            vmin
            vmax
            component
            unused_params

            """
            acqu_data, demag_data = self.mobj.equal_acqu_demag_steps(vmin=vmin, vmax=vmax)

            slope, sigma, yint, xint = lin_regress(pdd=acqu_data, column_name_x=component,
                                                   ypdd=demag_data, column_name_y=component)

            self.set_result(slope, 'slope')
            self.set_result(sigma, 'sigma')
            self.set_result(yint, 'yint')
            self.set_result(xint, 'xint')
            self.set_result(len(acqu_data), 'n')
            # self.mobj.sobj.results.loc[self.mobj.mid, 'slope'] = slope
            # self.mobj.sobj.results.loc[self.mobj.mid, 'sigma'] = sigma
            # self.mobj.sobj.results.loc[self.mobj.mid, 'yint'] = yint
            # self.mobj.sobj.results.loc[self.mobj.mid, 'xint'] = xint
            # self.mobj.sobj.results.loc[self.mobj.mid, 'n'] = len(acqu_data)

    class sigma(slope): pass

    class yint(slope): pass

    class xint(slope): pass

    class n(slope): pass

    class banc(Result):
        dependencies = ('slope', 'sigma')
        def recipe_default(self, vmin=20, vmax=700, component='m', blab=35.0,
                           **unused_params):
            """
            calculates the :math:`B_{anc}` value for a given lab field in the specified temperature interval.
            

            Parameters
            ----------
                vmin: float
                    min variable for best line fit
                vmax:float
                    max variable for best line fit
                component: str
                    component to be used for best line fit
                blab: lab field
                unused_params: dict
                    anything that is passed to another result class
                            
            Note
            ----
                This calculation method calls calculate_slope if you call it again afterwards, with different
                calculation_parameters, it will not influence this result. Therfore you have to be careful when calling
                this.

            """
            slope = self.mobj.sobj.results.loc[self.mobj.mid, 'slope']
            sigma = self.mobj.sobj.results.loc[self.mobj.mid, 'sigma']

            self.mobj.sobj.results.loc[self.mobj.mid, 'banc'] = abs(blab * slope)
            self.mobj.sobj.results.loc[self.mobj.mid, 'sigma_banc'] = abs(blab * sigma)

    class sigma_banc(banc):
        dependencies = ('slope', 'banc')
        indirect = True

    class f(slope):
        dependencies = ('slope', 'yint')
        def recipe_default(self, vmin=20, vmax=700, component='m', **unused_params):
            """
    
            The remanence fraction, f, was defined by Coe et al. (1978) as:
    
            .. math::
    
               f =  \\frac{\\Delta y^T}{y_0}
    
            where :math:`\Delta y^T` is the length of the NRM/TRM segment used in the slope calculation.
    
    
            :param parameter:
            :return:
    
            """
            delta_y_dash = self.delta_y_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            y_int = self.get_result('yint')

            self.set_result(result=delta_y_dash / abs(y_int), result_name='f')

    ####################################################################################################################
    """ F_VDS """
    class fvds(slope):
        dependencies = ('slope', )
        def recipe_default(self, vmin=20, vmax=700, component='m',
                           **unused_params):
            """
    
            NRM fraction used for the best-fit on an Arai diagram calculated as a vector difference sum (Tauxe and Staudigel, 2004).
    
            .. math::
    
               f_{VDS}=\\frac{\Delta{y'}}{VDS}
    
            Parameters
            ----------
            vmin
            vmax
            component
            unused_params
    
            """

            delta_y = self.delta_y_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            VDS = self.vds(vmin, vmax)
            self.set_result(result=delta_y / VDS, result_name='fvds')

    ####################################################################################################################
    """ FRAC """

    class frac(slope):
        def recipe_default(self, vmin=20, vmax=700, **unused_params):
            """
    
            NRM fraction used for the best-fit on an Arai diagram determined entirely by vector difference sum
            calculation (Shaar and Tauxe, 2013).
    
            .. math::
    
                FRAC=\\frac{\sum\limits_{i=start}^{end-1}{ \left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right| }}{VDS}
    
            :param parameter:
            :return:
    
            """
            NRM_sum = np.sum(np.linalg.norm(self.vd(vmin=vmin, vmax=vmax, **unused_params), axis=1))
            VDS = self.vds(vmin, vmax=vmax)
            self.set_result(result=NRM_sum / VDS, result_name='frac')

    ####################################################################################################################
    """ BETA """

    class beta(Result):
        dependencies = ('slope','sigma')

        def recipe_default(self, vmin=20, vmax=700, component='m',
                           **unused_params):
            """
    
            :math:`\beta` is a measure of the relative data scatter around the best-fit line and is the ratio of the
            standard error of the slope to the absolute value of the slope (Coe et al., 1978)
    
            .. math::
    
               \\beta = \\frac{\sigma_b}{|b|}
    
    
            :param parameters:
            :return:
    
            """

            slope = self.get_result('slope')
            sigma = self.get_result('sigma')
            result = sigma / abs(slope)
            self.set_result(result)

    ####################################################################################################################
    """ G """

    class g(slope):
        dependencies = ['slope']
        def recipe_default(self, vmin=20, vmax=700, component='m',
                        **unused_params):
            """
    
            Gap factor: A measure of the gap between the points in the chosen segment of the Arai plot and the least-squares
            line. :math:`g` approaches :math:`(n-2)/(n-1)` (close to unity) as the points are evenly distributed.
    
            """
            y_dash = self.y_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            delta_y_dash = self.delta_y_dash(vmin=vmin, vmax=vmax, component=component, **unused_params)
            y_dash_diff = [(y_dash[i + 1] - y_dash[i]) ** 2 for i in range(len(y_dash) - 1)]
            y_sum_dash_diff_sq = np.sum(y_dash_diff, axis=0)

            result = 1 - y_sum_dash_diff_sq / delta_y_dash ** 2
            self.set_result(result, 'g')


    ####################################################################################################################
    """ GAP MAX """
    class gapmax(slope):
        def recipe_default(self, vmin=20, vmax=700, **unused_params):
            """
            The gap factor is a measure of the average Arai plot point spacing and may not represent extremes
            of spacing. To account for this Shaar and Tauxe (2013)) proposed :math:`GAP_{\text{MAX}}`, which is the maximum
            gap between two points determined by vector arithmetic.
    
            .. math::
               GAP_{\\text{MAX}}=\\frac{\\max{\{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|\}}_{i=start, \\ldots, end-1}}
               {\\sum\\limits_{i=start}^{end-1}{\\left|\\mathbf{NRM}_{i+1}-\\mathbf{NRM}_{i}\\right|}}
    
            :return:
    
            """
            vd = self.vd(vmin=vmin, vmax=vmax)
            vd = np.linalg.norm(vd, axis=1)
            max_vd = np.max(vd)
            sum_vd = np.sum(vd)
            result =  max_vd / sum_vd
            self.set_result(result)

    ####################################################################################################################
    """ Q """

    class q(slope):
        dependencies = ['beta', 'f', 'g']

        def recipe_default(self, vmin=20, vmax=700, component='m', **unused_params):
            """
            The quality factor (:math:`q`) is a measure of the overall quality of the paleointensity estimate and combines
            the relative scatter of the best-fit line, the NRM fraction and the gap factor (Coe et al., 1978).
    
            .. math::
               q=\\frac{\\left|b\\right|fg}{\\sigma_b}=\\frac{fg}{\\beta}
    
            :param parameter:
            :return:
    
            """
            beta = self.get_result('beta')
            f = self.get_result('f')
            gap = self.get_result('g')
            result = (f * gap) / beta
            self.set_result(result)

    ####################################################################################################################
    """ W """

    class w(Result):
        dependencies =  ('q','n')

        def recipe_default(self, vmin=20, vmax=700, component='m', **unused_params):
            """
            Weighting factor of Prevot et al. (1985). It is calculated by

            .. math::

               w=\\frac{q}{\\sqrt{n-2}}

            Originally it is :math:`w=\\frac{fg}{s}`, where :math:`s^2` is given by

            .. math::

               s^2 = 2+\\frac{2\\sum\\limits_{i=start}^{end}{(x_i-\\bar{x})(y_i-\\bar{y})}}
                  {\\left( \\sum\\limits_{i=start}^{end}{(x_i- \\bar{x})^{\\frac{1}{2}}}
                  \\sum\\limits_{i=start}^{end}{(y_i-\\bar{y})^2} \\right)^2}

            It can be noted, however, that :math:`w` can be more readily calculated as:

            .. math::

               w=\\frac{q}{\\sqrt{n-2}}

            :param parameter:
            """
            q = self.get_result('q')
            n = self.get_result('n')
            result = q / np.sqrt((n - 2))
            self.set_result(result)


if __name__ == '__main__':
    s = RockPy.Sample('61')
    # m = s.add_measurement(mtype='paleointensity',
    #                       ftype='jr6',
    #                       fpath=os.path.join(RockPy.test_data_path, 'TT-paleointensity.jr6'),
    #                       dialect='tdt')
    #
    # print(m.data)

    m = s.add_simulation(mtype='pint', preset='Fabian7a', pressure_demag= True, d2=0.2, d3=0.4, dt=0.2)
    m.calc_all(vmin=200)
    # m.ftype_data.plot_arai()
    m.banc(vmin=200)
    plt.show()
