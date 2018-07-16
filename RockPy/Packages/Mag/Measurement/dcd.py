__author__ = 'volk'
import RockPy
import logging
from RockPy.core.measurement import Measurement
from RockPy.core.result import Result
import numpy as np
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

    @property
    def log_data(self):
        out = super().data
        out = out.set_index('B')
        out.index = np.log(-out.index)
        return out

    #################################################################################
    # ABSTRACT METHODS
    #################################################################################
    def set_initial_state(self, mtype=None, fpath=None, ftype=None, mobj=None, series=None):
        pass

    def set_calibration_measurement(self, fpath=None, mdata=None, mobj=None):
        pass

    def delete_dtype_var_val(self, dtype, var, val):
        pass

    ####################################################################################################################
    """ formatting functions """

    @staticmethod
    def format_vsm(ftype_data, **kwargs):
        """
        formatting routine for VSM type hysteresis measurements. Indirectly called by the measurement.from_file method.

        Parameters
        ----------
        ftype_data: RockPy.io
            data read by Mag.io.Vsm.vsm function.
            Contains:
              - data
              - header
              - segment_header


        Returns
        -------
            data: pandas.Dataframe
                columns = ['B','M']
            ftype_data: io object
                io object as read by Mag.io.Vsm.vsm


        """

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['Field (T)', 'Remanence (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            Dcd.log().error('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        segment_index = ftype_data.mtype.index('dcd')
        data = ftype_data.get_segment_data(segment_index).rename(columns={"Field (T)": "B", "Remanence (Am2)": "M"})

        return data

    @staticmethod
    def format_vftb(ftype_data, sobj_name=None):  # todo implement VFTB
        """
        formats the output from vftb to measurement.data
        :return:
        """
        raise NotImplementedError

    ####################################################################################################################
    ''' Mrs '''

    class Mrs(Result):
        def recipe_max(self, **non_method_parameters):
            """
            Magnetic Moment at first measurement point
            """
            m = self.mobj
            result = m.data['M'].max()
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.array(result)

    ####################################################################################################################
    ''' Bcr '''

    class Bcr(Result):
        """
        Calculates the coercivity of remanence from the dcd curve
        """
        default_recipe = 'nonlinear'

        def recipe_linear(self, npoints=4, check=False):
            """
            Calculates the coercivity using a linear interpolation between the points crossing the x axis for upfield
            and down field slope.

            Parameters
            ----------
                check: bool
                    creates a small plot to check results
                npoints: int
                    number of points to use for fit

            Note
            ----
                Uses scipy.linregress for calculation
            """
            self.recipe_nonlinear(npoints=npoints, order=1, check=check)

        def recipe_nonlinear(self, npoints=4, order=2, check=False):
            """
            Calculates the coercivity of remanence using a spline interpolation between the points crossing
            the x axis for upfield and down field slope.

            Parameters
            ----------
                check: bool
                    creates a small plot to check results
                npoints: int
                    default: 4
                    number of points to use for fit
                order: int
                    default: 2
                    order of polynomial fit

            Note
            ----
                Uses numpy.polyfit for calculation
            """

            # raise NotImplementedError
            m = self.mobj

            if npoints > len(m.data):
                npoints = len(m.data) - 1

            # get magnetization limits for a calculation using the n points closest to 0
            moment = sorted(abs(m.data['M'].values))[npoints - 1]

            # filter data for fields higher than field_limit
            data = m.data[m.data['M'].abs() <= moment]

            # fit second order polynomial
            fit = np.polyfit(data['M'].values, data.index, order)
            result = np.poly1d(fit)(0)

            if check:
                y = np.linspace(data['M'].values[0], data['M'].values[-1])
                x_new = np.poly1d(fit)(y)

                plt.plot(-data.index, data['M'], '.', color=RockPy.colors[0], mfc='w',label='data')
                plt.plot(-x_new, y, color=RockPy.colors[0], label='fit')
                plt.plot(-result, 0, 'xk',label='B$_{cr}$')
                plt.axhline(0, color = 'k', zorder=0)

                plt.gca().text(0.05, 0.1, 'B$_{cr}$ = %.2f mT'%(abs(result)*1000),
                           verticalalignment='bottom', horizontalalignment='left',
                           transform=plt.gca().transAxes,
                           bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0),
                           color='k')

                plt.xlabel('B [mT]')
                plt.ylabel('M [Am$^2$]')
                plt.legend(frameon=True)
                plt.grid()
                plt.show()

            # set result so it can be accessed
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.array(result)

if __name__ == '__main__':

    S = RockPy.Study()
    s = S.add_sample('FeCoAa36-G02')
    m = s.add_measurement('/Users/mike/Dropbox/github/collaborations/Cournede (IRM)/data/VSM/FeCo_FeCoAa36-G02_(DCD,IRM)_VSM#61.9mg.001')
