import logging
import RockPy
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

from RockPy.core.measurement import Measurement
from RockPy.core.result import Result
from RockPy.core.utils import correction
from RockPy.tools.compute import lin_regress
import RockPy.Packages.Magnetism.simulations
from RockPy.tools.pandas_tools import get_values_in_both

class Hysteresis(Measurement):

    @property
    def data(self):
        out = super().data
        return out.set_index('B')

    ####################################################################################################################
    """ formatting functions """

    @staticmethod
    def _format_vsm(ftype_data, sobj_name=None):
        '''
        formatting routine for VSM type hysteresis measurements. Indirectly called by the measurement.from_file method.

        Parameters
        ----------
        ftype_data: RockPy.io
            data read by Magnetism.Ftypes.Vsm.vsm function.
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
            ftype_data: Ftypes object
                Ftypes object as read by Magnetism.Ftypes.Vsm.vsm


        '''

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['Field (T)', 'Moment (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            Hysteresis.log().debug('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        data = ftype_data.data.rename(columns={"Field (T)": "B", "Moment (Am2)": "M"})
        data = data.dropna(how='all')
        return data

    @staticmethod
    def _format_agm(ftype_data, sobj_name=None):
        return Hysteresis._format_vsm(ftype_data, sobj_name)

    @staticmethod
    def _format_vftb(ftype_data, sobj_name=None):
        '''
        formatting routine for VSM type hysteresis measurements. Indirectly called by the measurement.from_file method.

        Parameters
        ----------
        ftype_data: RockPy.io
            data read by Magnetism.Ftypes.Vsm.vsm function.
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
            ftype_data: Ftypes object
                Ftypes object as read by Magnetism.Ftypes.Vsm.vsm


        '''

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['B', 'M']
        data = ftype_data.data
        data = data.dropna(how='all')
        return data
    ####################################################################################################################
    ''' BRANCHES '''

    @property
    def fieldspacing(self):
        """
        Property returns the mean absolute spacing between field steps

        Returns
        -------
            float
        """
        return np.mean(np.abs(np.diff(self.data.index)))

    @property
    def max_field(self):
        """ returns maximum field of measurement """
        return np.nanmax(np.abs(self.data.index.values))

    @property
    def _regularize_fields(self):
        '''
        Method generates new field steps using the mean field spacing and the maximum applied field, with regular steps
        Returns
        -------
            np.array with reugularized field steps
        '''
        virgin_fields = np.arange(0, self.max_field + self.fieldspacing, self.fieldspacing)
        df_fields = np.arange(self.max_field, (self.max_field + self.fieldspacing) * -1, -self.fieldspacing)
        uf_fields = np.arange(-1 * self.max_field, self.max_field + self.fieldspacing, self.fieldspacing)
        if self.has_virgin:
            fields = np.concatenate([virgin_fields, df_fields, uf_fields])
        else:
            fields = np.concatenate([df_fields, uf_fields])
        return np.round(fields, 2)

    def has_virgin(self):
        '''
        Method to determine weather a virgin/MSi branch was measured.

        Checks if the first field step is > 90% of the max field. If yes than it is no virgin/MSi

        Returns
        -------
            bool
        '''
        # check if the first point is  close to the maximum/minimum field
        if np.abs(self.data.index[0]) > 0.9 * self.data.index.max():
            return False
        else:
            return True

    def get_polarity_switch(self, window=5):
        '''

        Parameters
        ----------
        window : int
            default_recipe: 1 - no running mean
            Size of the moving window. This is the number of observations used for calculating the statistic.
            Each window will be a fixed size. Moving window is used twice: for smoothing data (mean with hamming window)
            and smoothing the sign (median, no special window)

        Notes
        -----
            A window size of 5 is usually sufficient to smooth out 1% of noise

        Returns
        -------
            pandas.Series with sign of the polarity
        '''
        a = self.data.index.to_series()

        # adding 2% noise to the data
        # a += np.random.normal(0,0.05*a.max(),a.size)
        # todo maybe add convergence with increasing window size for automatic smoothing

        if window > 1:
            a = a.rolling(window, win_type='hamming', center=True).mean()

        # calculating differences between individual points
        diffs = a.diff()
        diffs = diffs.rolling(window, center=True).median()

        # filling in missing data due to window size
        diffs = diffs.fillna(method='bfill')  # filling missing values at beginning
        diffs = diffs.fillna(method='ffill')  # filling missing values at end

        # reduce to sign of the differences
        asign = diffs.apply(np.sign)

        return asign

    def get_polarity_switch_index(self, window=1):
        '''
        Method calls hysteresis.get_polarity_switch with window and then calculated the indices of the switch

        Parameters
        ----------
        window: int
            default_recipe: 1 - no running mean
            Size of the moving window. This is the number of observations used for calculating the statistic.

        Returns
        -------
            np.array of indices
        '''

        asign = self.get_polarity_switch()

        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

        # return np.where(signchange!=0)[0].astype(int)
        return (np.where(signchange != 0)[0]).astype(int)

    @property
    def downfield(self):
        '''

        Returns
        -------
            pandas.DataFrame with only downfield data. Window size for selecting the polarity change is 5
        '''
        # todo how to change window size?
        idx = self.get_polarity_switch_index(5)
        if len(idx) > 1:
            return self.data.iloc[int(idx[0]):int(idx[1])].dropna(axis=1).dropna()
        else:
            return self.data.iloc[0:int(idx[1])].dropna(axis=1).dropna()

    @property
    def upfield(self):
        '''

        Returns
        -------
            pandas.DataFrame with only upfield data. Window size for selecting the polarity change is 5
        '''
        # todo how to change window size?
        idx = self.get_polarity_switch_index(5)
        return self.data.iloc[int(idx[-1]) - 1:].dropna(axis=1).dropna()

    """ CALCULATIONS """

    @property
    def irreversible(self):
        """
        Calculates the irreversible hysteretic components :math:`M_{ih}` from the data.

        .. math::

           M_{ih} = (M^+(H) + M^-(H)) / 2

        where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop

        Returns
        -------
           Mih: RockPyData

        """

        raise NotImplementedError

    def get_reversible(self):  # todo implement reversible part
        # """
        # Calculates the reversible hysteretic components :math:`M_{rh}` from the data.
        #
        # .. math::
        #
        #    M_{ih} = (M^+(H) - M^-(H)) / 2
        #
        # where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop
        #
        # Returns
        # -------
        #    Mrh: RockPyData
        #
        # """
        raise NotImplementedError

    """ RESULTS """
    ####################################################################################################################
    """ BC """

    class Bc(Result):
        default_recipe = 'linear'

        def recipe_linear(self, npoints=4, check=False):
            """
            Calculates the coercivity using a linear interpolation between the points crossing the x axis for upfield and down field slope.

            Parameters
            ----------
                field_limit: float
                    default_recipe: 0, 0mT
                    the maximum/ minimum fields used for the linear regression

            Note
            ----
                Uses numpy.polyfit for calculation
            """
            self.recipe_nonlinear(npoints=npoints, order=1, check=check)

        def recipe_nonlinear(self, npoints=4, order=2, check=False):
            """
            Calculates the coercivity using a spline interpolation between the points crossing
            the x axis for upfield and down field slope.

            Parameters
            ----------
                npoints: int
                    default_recipe: 4
                    number of points used for fit

            Note
            ----
                Uses numpy.polyfit for calculation
            """
            # retireve measurement instance
            m = self.mobj

            # initialize result
            result = []

            # the field_limit has to be set higher than the lowest field
            # if not the field_limit will be chosen to be 2 points for uf and df separately

            m = self.mobj

            if npoints > len(m.data):
                npoints = len(m.data) - 1

            if npoints < 2:
                self.log().warning('NPOINTS INCOMPATIBLE minimum 2 required')
                self.log().warning('\t\t setting NPOINTS - << 2 >> ')
                npoints = 2
                self.params['npoints'] = npoints
            if npoints > len(m.downfield):
                self.log().warning('NPOINTS INCOMPATIBLE maximum %i allowed' % (len(m.downfield)))
                self.log().warning('\t\t setting NPOINTS - << %i >> ' % (len(m.downfield)))
                npoints = len(m.downfield)
                self.params['npoints'] = npoints

            # get magnetization limits for a calculation using the n points closest to 0
            moment = sorted(abs(m.data['M'].values))[npoints - 1]

            # get magnetization limits for a calculation using the n points closest to 0
            df_moment = sorted(abs(m.downfield['M'].values))[npoints - 1]
            uf_moment = sorted(abs(m.upfield['M'].values))[npoints - 1]

            # filter data for fields higher than field_limit
            df_data = m.downfield[m.downfield['M'].abs() <= df_moment]
            uf_data = m.upfield[m.upfield['M'].abs() <= uf_moment]

            # fit polynomial
            df_fit = np.polyfit(df_data['M'].values, df_data.index, order)
            uf_fit = np.polyfit(uf_data['M'].values, uf_data.index, order)
            result = [np.poly1d(df_fit)(0), np.poly1d(uf_fit)(0)]

            if check:
                ''' upper '''
                l, = plt.plot(-df_data.index, -df_data['M'], '.', mfc='w', label='%s data' % ('upper'))
                y = np.linspace(df_data['M'].iloc[0], df_data['M'].iloc[-1])
                plt.plot(-np.poly1d(df_fit)(y), -y, '--', label='%s fit' % ('upper'), color=l.get_color())
                ''' lower '''
                l, = plt.plot(uf_data.index, uf_data['M'], '.', mfc='w', label='%s data' % ('lower'))
                y = np.linspace(uf_data['M'].iloc[0], uf_data['M'].iloc[-1])
                plt.plot(np.poly1d(uf_fit)(y), y, '--', label='%s fit' % ('upper'), color=l.get_color())

                plt.plot(np.abs(result), [0, 0], 'ko', mfc='none', label='Bc(branch)')
                plt.plot(np.nanmean(np.abs(result)), 0, 'xk', label='mean Bc')
                plt.grid()
                plt.xlabel('B [T]')
                plt.ylabel('M [Am$^2$]')
                plt.title('Bc - check')
                plt.text(0.8, 0.1, '$B_c = $ %.1f mT' % (np.nanmean(np.abs(result)) * 1000),
                         transform=plt.gca().transAxes,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

                plt.legend()
                plt.show()

            result = np.abs(result)
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nanmean(result)

    ####################################################################################################################
    """ MRS """

    class Mrs(Result):
        def recipe_default(self, npoints=4, check=False, **unused_params):

            # set measurement instance
            m = self.mobj
            # initialize result
            result = []

            # get magnetization limits for a calculation using the n points closest to 0 fro each direction
            df_moment = sorted(abs(m.downfield['M'].values))[npoints - 1]
            uf_moment = sorted(abs(m.upfield['M'].values))[npoints - 1]

            # filter data for fields higher than field_limit
            down_f = m.downfield[m.downfield['M'].abs() <= df_moment]
            up_f = m.upfield[m.upfield['M'].abs() <= uf_moment]

            for i, dir in enumerate([down_f, up_f]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(dir.index, dir['M'])
                result.append(intercept)

                # check plot
                if check:
                    x = np.linspace(dir.index.min(), dir.index.max(), 100)
                    y_new = slope * x + intercept
                    l, = plt.plot(dir.index, dir['M'], '.')
                    plt.plot(x, y_new, '--', color=l.get_color())

            # check plot
            if check:
                plt.plot([0, 0], result, 'ko', mfc='w', label='$M_{rs}$ (branch)')
                plt.plot([0, 0], [-np.nanmean(np.abs(result)), np.nanmean(np.abs(result))], 'xk', label='mean $M_{rs}$')
                plt.grid()
                plt.xlabel('Field')
                plt.ylabel('Moment')
                plt.title('$M_{rs}$ - check')
                plt.legend()
                plt.text(0.8, 0.1, '$M_{rs} = $ %.1f ' % (np.nanmean(np.abs(result))),
                         transform=plt.gca().transAxes,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

                plt.tight_layout()
                plt.show()

            result = np.abs(result)
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nanmean(result)

    ####################################################################################################################
    """ MS """

    class Ms(Result):
        """
        Calculates the saturation magnetization from a hysteresis loop. The standard recipe is 'linear'.

        Recipes
        -------
            - simple:
                simple linear fit for fields higher than specified
            - app2sat:
                Uses approach to saturation to calculate Ms, Hf_sus

        Parameters
        ----------
            recipe = simple
            ---------------
                saturation_percent: float
                    percent where saturation is assumed
                    default = 75.
                ommit_last_n: int
                    omits the last n points
                    default = 0
                check: bool
                    creates a plot to check the result
                    default:False

            recipe = app2sat
            ---------------
                saturation_percent: float
                    percent where saturation is assumed
                    default = 75.
                ommit_last_n: int
                    omits the last n points
                    default = 0
                check: bool
                    creates a plot to check the result
                    default:False



        """

        default_recipe = 'simple'

        @staticmethod
        def approach2sat_func(h, ms, chi, alpha, beta=-2):
            """
            General approach to saturation function

            Parameters
            ----------
               x: ndarray
                  field
               ms: float
                  saturation magnetization
               chi: float
                  susceptibility
               alpha: float
               beta: float
                  not fitted assumed -2

            Returns
            -------
               ndarray:
                  :math:`M_s \chi * B + \\alpha * B^{\\beta = -2}`
            """
            return ms + chi * h + alpha * h ** beta

        def get_df_uf_plus_minus(self, saturation_percent, ommit_last_n):
            """
            Filters the data :code:`down_field`, :code:`up_field` to be larger than the saturation_field, filters
            the last :code:`ommit_last_n` and splits into pos and negative components
            """
            # transform from percent value
            saturation_percent /= 100

            # filter ommitted points
            if ommit_last_n > 0:
                df = self.mobj.downfield.iloc[ommit_last_n:-ommit_last_n]
                uf = self.mobj.upfield.iloc[ommit_last_n:-ommit_last_n]
            else:
                df = self.mobj.downfield
                uf = self.mobj.upfield

            # filter for field limits
            df_plus = df[df.index >= saturation_percent * self.mobj.max_field]
            df_minus = df[df.index <= -(saturation_percent * self.mobj.max_field)]

            uf_plus = uf[uf.index >= saturation_percent * self.mobj.max_field]
            uf_minus = uf[uf.index <= -(saturation_percent * self.mobj.max_field)]

            return df_plus, df_minus, uf_plus, uf_minus

        def recipe_app2sat(self, saturation_percent=75., ommit_last_n=0, check=False):
            """
            Calculates the high field susceptibility and Ms using approach to saturation
            :return:
            """
            df_pos, df_neg, uf_pos, uf_neg = self.get_df_uf_plus_minus(saturation_percent=saturation_percent,
                                                                       ommit_last_n=ommit_last_n)

            # initialize out
            ms = []
            slope = []
            alpha = []

            for d in [df_pos, df_neg, uf_pos, uf_neg]:
                fields = d.index
                moments = d['M'].values

                if len(fields) < 2:
                    self.log.warning('CANT calculate approach to saturation. Not enough points (<=2) in data. '
                                     'Consider using smaller <saturation_percent> value')
                    continue
                popt, pcov = curve_fit(self.approach2sat_func, np.fabs(fields), moments * np.sign(np.mean(fields)),
                                       p0=[max(abs(moments)) / 2, 0, 0])

                ms.append(popt[0])
                slope.append(popt[1])
                alpha.append(popt[2])

            self.mobj.sobj.results.loc[self.mobj.mid, 'Hf_sus'] = np.nanmean(slope)
            self.mobj.sobj.results.loc[self.mobj.mid, 'Ms'] = np.nanmean(np.abs(ms))
            self.mobj.sobj.results.loc[self.mobj.mid, 'alpha'] = np.nanmean(alpha)

            if check:

                for i, d in enumerate([df_pos, df_neg, uf_pos, uf_neg]):
                    fields = np.abs(d.index)
                    moments = d['M'].values

                    # calculate for raw data plot
                    raw_d = self.get_df_uf_plus_minus(saturation_percent=0, ommit_last_n=ommit_last_n)
                    # plot all data
                    plt.plot(np.abs(raw_d[i].index), raw_d[i]['M'] * np.sign(np.mean(raw_d[i]['M'].index)), '.',
                             mfc='w',
                             color=RockPy.colors[i], alpha=0.5, label='')
                    # plot data used for fit
                    plt.plot(fields, moments * np.sign(np.mean(raw_d[i]['M'].index)), '.', color=RockPy.colors[i],
                             label=['upper +', 'upper -', 'lower +', 'lower -'][i] + '(data)')
                    # plot app2sat function
                    plt.plot(np.linspace(0.1, max(fields)),
                             self.approach2sat_func(np.linspace(0.1, max(fields)), ms[i], slope[i], alpha[i], -2), '--',
                             color=RockPy.colors[i], label=['upper +', 'upper -', 'lower +', 'lower -'][i] + '(fit)')
                    # plot linear fit
                    plt.plot(np.linspace(0, max(fields)), slope[i] * np.linspace(0, max(fields)) + ms[i], '-',
                             color=RockPy.colors[i])

                plt.errorbar(0, np.mean(ms), yerr=2 * np.std(ms), color='k', marker='.', label='mean Ms (2$\sigma$)',
                             zorder=100, )
                plt.legend()
                plt.xlim(-0.01, max(fields))

                # plt.ylim(0, max(ms) * 1.1)
                plt.xlabel('B [T]')
                plt.ylabel('M [Am$^2$]')
                plt.show()

        def recipe_simple(self, saturation_percent=75., ommit_last_n=0, check=False):
            """
            Calculates High-Field susceptibility using a simple linear regression on all branches

            Parameters
            ----------
                saturation_percent: float
                    default_recipe: 75.0
                    Defines the field limit in percent of max(field) at which saturation is assumed.
                     e.g. max field : 1T -> saturation asumed at 750mT
                ommit_last_n: int, pos
                    last n points of each branch are not used for the calculation


            Calculation
            -----------

                calculates the slope using SciPy.linregress for each branch at positive and negative fields.
                Giving four values for the slope.
                The result is the mean for all four values, and the error is the standard deviation

            """

            # initialize result
            hf_sus_result = []
            ms_result = []

            if saturation_percent >= 100:
                self.log().warning('SATURATION > 100%! setting to default_recipe value (75%)')
                saturation_percent = 75.0

            df_plus, df_minus, uf_plus, uf_minus = self.get_df_uf_plus_minus(saturation_percent=saturation_percent,
                                                                             ommit_last_n=ommit_last_n)

            # calculate for each branch for positive and negative fields
            for i, dir in enumerate([df_plus, df_minus, uf_plus, uf_minus]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(dir.index, dir['M'])
                hf_sus_result.append(slope)
                ms_result.append(intercept)

                # check plot
                if check:
                    d0 = self.get_df_uf_plus_minus(0, 0)
                    x = np.linspace(0, self.mobj.max_field)
                    y_new = slope * x + abs(intercept)
                    l, = plt.plot(abs(d0[i].index), d0[i]['M'] * np.sign(np.mean(d0[i].index)), '.', mfc='w',
                                  label=['df+', 'df-', 'uf+', 'uf-'][i],
                                  color=RockPy.colors[i])
                    plt.plot(x, y_new, '--', color=l.get_color())

            # check plot
            if check:
                # plt.plot([0,0,0,0], np.abs(ms_result), 'ko', label='Ms (branch)', mfc='none', mew=0.5)
                plt.errorbar([0], np.mean(np.abs(ms_result)), yerr=2 * np.std(np.abs(ms_result)),
                             color='k', marker='.', label='mean Ms (2$\sigma$)', zorder=100, )
                plt.axvline(self.mobj.max_field * saturation_percent / 100, ls='--', color='grey')
                plt.xlabel('B [T]')
                plt.ylabel('M [Am$^2$}')
                plt.xlim([-self.mobj.max_field * 0.01, self.mobj.max_field])
                plt.legend()
                plt.grid()
                plt.show()

            self.mobj.sobj.results.loc[self.mobj.mid, 'Hf_sus'] = np.nanmean(hf_sus_result)
            self.mobj.sobj.results.loc[self.mobj.mid, 'Ms'] = np.nanmean(np.abs(ms_result))

    class Hf_sus(Ms):
        dependencies = ['ms']

    ####################################################################################################################
    ''' CORRECTIONS '''

    def rotate_branch(self, branch):
        """
        rotates a branch by 180 degrees, by multiplying the field and mag values by -1.

        Parameters
        ----------
            branch: str or. pandas.DataFrame
                up-field or down-field
                RockPyData: will rotate the data

        Returns
        -------
            deepcopy of branch
        """
        if isinstance(branch, str):
            data = getattr(self, branch).copy()

        if isinstance(branch, pd.DataFrame):
            data = branch.copy()

        data.index *= -1
        data['M'] *= -1

        return data.sort_index()  # todo may fail for hysteresis loops with virgin branch

    @classmethod
    def get_grid(cls, bmax=1, grid_points=30, tuning=10):
        """
        Creates a grid of field values

        Parameters
        ----------
        bmax
        grid_points
        tuning

        Returns
        -------

        """
        grid = []
        # calculating the grid
        for i in range(-grid_points, grid_points + 1):
            if i != 0:
                boi = (abs(i) / i) * (bmax / tuning) * ((tuning + 1) ** (abs(i) / float(grid_points)) - 1.)
            else:  # catch exception for i = 0
                boi = 0
            grid.append(boi)
        return np.array(grid)

    @correction
    def data_gridding(self, order=2, grid_points=20, tuning=1, ommit_n_points=0, check=False, **parameter):
        """
        Data griding after :cite:`Dobeneck1996a`. Generates an interpolated hysteresis loop with
        :math:`M^{\pm}_{sam}(B^{\pm}_{exp})` at mathematically defined (grid) field values, identical for upper
        and lower branch.

        .. math::

           B_{\text{grid}}(i) = \\frac{|i|}{i} \\frac{B_m}{\lambda} \\left[(\lambda + 1 )^{|i|/n} - 1 \\right]

        Parameters
        ----------

           method: str
              method with which the data is fitted between grid points.

              first:
                  data is fitted using a first order polinomial :math:`M(B) = a_1 + a2*B`
              second:
                  data is fitted using a second order polinomial :math:`M(B) = a_1 + a2*B +a3*B^2`

           parameter: dict
              Keyword arguments passed through

        See Also
        --------
           get_grid
        """

        if any([len(i.index) <= 50 for i in [self.downfield, self.upfield]]):
            self.log.warning('Hysteresis branches have less than 50 (%i) points, gridding not possible' % (
                len(self.data['down_field']['field'].v)))
            return

        bmax = min([max(self.downfield.index), max(self.upfield.index)])
        bmin = max([min(self.downfield.index), min(self.upfield.index)])
        bm = max([abs(bmax), abs(bmin)])

        grid = self.get_grid(bmax=bm, grid_points=grid_points, tuning=tuning, **parameter)

        # interpolate the magnetization values M_int(Bgrid(i)) for i = -n+1 .. n-1
        # by fitting M_{measured}(B_{experimental}) individually in all intervals [Bgrid(i-1), Bgrid(i+1)]
        # with first or second order polinomials

        if check:
            uncorrected_data = self.data.copy()

        # initialize DataFrame for gridded data
        interp_data = pd.DataFrame(columns=self.data.columns)

        for n, dtype in enumerate(['downfield', 'upfield', 'virgin']):
            aux = pd.DataFrame(columns=self.data.columns)

            # catch missing branches
            if not hasattr(self, dtype):
                continue

            d = getattr(self, dtype)

            if ommit_n_points > 0:
                d = d.iloc[ommit_n_points:-ommit_n_points]
            # cycle through gridpoints

            for col in self.data.columns:
                # calculate interpolation for all columns (e.g. temperature)
                if col in ['B', 'sID', 'mID']:
                    continue

                for i, B in enumerate(grid):
                    # set B to B column
                    aux.loc[i, 'B'] = B

                    # indices of points within the grid points
                    if i == 0:
                        idx = [j for j, v in enumerate(d.index) if v <= grid[i]]
                    elif i == len(grid) - 1:
                        idx = [j for j, v in enumerate(d.index) if grid[i] <= v]
                    else:
                        idx = [j for j, v in enumerate(d.index) if grid[i - 1] <= v <= grid[i + 1]]

                    if len(idx) > 1:  # if no points between gridpoints -> no interpolation
                        data = d.iloc[idx]

                        # make fit object
                        fit = np.polyfit(data.index, data[col], order)
                        # calculate Moment at B
                        dfit = np.poly1d(fit)(B)

                        aux.loc[i, col] = dfit

                        # plt.plot(data.index, data[col], '.')
                        # plt.plot(np.linspace(min(data.index), max(data.index)),
                        #          np.poly1d(fit)(np.linspace(min(data.index), max(data.index))), '-')
                        # plt.plot(B, dfit, 'x')
                        # plt.show()

                # set dtype to float -> calculations dont work -> pandas sets object
                aux[col] = aux[col].astype(np.float)

            if dtype == 'downfield':
                aux = aux.iloc[::-1]
                aux = aux.reset_index(drop=True)

            interp_data = pd.concat([interp_data, aux], sort=True)
            # interp_data.index = interp_data.index.astype(np.float)

        self.replace_data(interp_data)

        if check:
            ax = self.check_plot(self.data, uncorrected_data)
            return ax

    # def correct_symmetry(self, check=False):
    #
    #     if check:
    #         uncorrected_data = self.data.copy()
    #
    #     df = self.downfield
    #     uf = self.upfield
    #
    #     uf_rotate = self.rotate_branch(uf)
    #     df_rotate = self.rotate_branch(df)
    #
    #     fields = sorted(list(set(df.index) | set(uf.index) | set(df_rotate.index) | set(uf_rotate.index)))
    #
    #     # interpolate all branches and rotations
    #     df = df.interpolate(fields)
    #     uf = uf.interpolate(fields)
    #
    #     # df_rotate = df_rotate.interpolate(fields)
    #     uf_rotate = uf_rotate.interpolate(fields)
    #
    #     down_field_corrected = deepcopy(df)
    #     up_field_corrected = deepcopy(uf)
    #     down_field_corrected['mag'] = (df['mag'].v + uf_rotate['mag'].v) / 2
    #
    #     up_field_corrected['field'] = - down_field_corrected['field'].v
    #     up_field_corrected['mag'] = - down_field_corrected['mag'].v
    #
    #     self.data.update(dict(up_field=up_field_corrected, down_field=down_field_corrected))
    #
    #     if check:
    #         self.check_plot(uncorrected_data=uncorrected_data)

    ####################################################################################################################
    ''' PLOTTING '''

    @staticmethod
    def check_plot(corrected_data, uncorrected_data, ax=None, f=None, points=None, show=True, title='', **kwargs):
        """
        Helper function for consistent check visualization

        Parameters
        ----------
           uncorrected_data: RockPyData
              the pre-correction data.
        """
        if not ax:
            f, ax = plt.subplots()

        ax.plot(uncorrected_data.index, uncorrected_data['M'], color='r', marker='o', ls='', mfc='none')
        ax.plot(corrected_data.index, corrected_data['M'], color='g', marker='.')

        if points:
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], marker='o', **kwargs)

        ax.set_ylabel('Moment')
        ax.set_xlabel('Field')
        ax.legend(['corrected / fitted', 'original'], loc='best')
        ax.grid(zorder=1)
        ax.set_title(title)
        ax.axhline(color='k', zorder=1)
        ax.axvline(color='k', zorder=1)
        ax.set_xlim([min(corrected_data.index), max(corrected_data.index)])
        plt.tight_layout()
        return ax

    def plot(self, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        ax.plot(self.data['M'])

class Paleointensity(Measurement):

    def equal_acqu_demag_steps(self, vmin=20, vmax=700):
        """
        Filters the th and ptrm data so that the temperatures are within vmin, vmax and only temperatures in both
        th and ptrm are returned.
        """

        # get equal temperature steps for both demagnetization and acquisition measurements
        equal_steps = get_values_in_both(self.zf_steps, self.if_steps, key='level')

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
            simobj = RockPy.Packages.Magnetism.simulations.Fabian2001(**simparams)

        return cls(sobj=sobj, mdata=simobj.get_data(pressure_demag=pressure_demag),
                   series=series,
                   ftype_data=None, simobj=simobj, ftype='simulation')

    @staticmethod
    def _format_jr6(ftype_data, sobj_name=None):
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
    def _format_tdt(ftype_data, sobj_name=None):
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

    @staticmethod
    def _format_cryomag(ftype_data, sobj_name):
        '''
        formats the data fro ftype.cryomag into paleointensity readable format

        Parameters
        ----------
        ftype_data
        sobj_name

        Returns
        -------

        '''
        # only use results
        data = ftype_data.data[ftype_data.data['mode'] == 'results']

        # rename the columns from magic format -> RockPy internal names
        data = data.rename(
            columns={'magn_x': 'x', 'magn_y': 'y', 'magn_z': 'z', 'magn_moment': 'm'})

        data['ti'] = data['level']

        for col in ('coreaz', 'coredip', 'bedaz', 'beddip', 'vol', 'weight', 'a95', 'Dc', 'Ic', 'Dg', 'Ig', 'Ds', 'Is'):
            del data[col]

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
        d = d.groupby(d.index).first()
        return d

    @property
    def nrm(self):
        d = self.data[self.data['LT_code'] == 'LT-NO'].set_index('ti')
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
        d = d.groupby(d.index).first()

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
        d = d.groupby(d.index).first()
        plt.Line2D
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
        pTRM acquisition steps of the experiments. Vector substration of the pt[[x,y,z]] and th[[x,y,z]] steps for
        each ti. Also recalculates the moment ['m'] using np.linalg.norm

        Returns
        -------
            pandas.DataFrame
        """

        equal_vals = get_values_in_both(self.if_steps, self.zf_steps, key = 'level')

        if_steps = self.if_steps[np.in1d(self.if_steps['level'], equal_vals)].copy()
        zf_steps = self.zf_steps[np.in1d(self.zf_steps['level'], equal_vals)].copy()

        try:
            if_steps.loc[:, ('x', 'y', 'z')] -= zf_steps.loc[:, ('x', 'y', 'z')]
        except ValueError:
            raise ValueError('cannot reindex from a duplicate axis -- likely duplicate values for IF or ZF steps\n'
                             'IF steps: %s \nZF_steps: %s' % (if_steps['level'].values, zf_steps['level'].values))
        if_steps['LT_code'] = 'PTRM'
        if_steps['m'] = np.sqrt(if_steps.loc[:, ['x', 'y', 'z']].apply(lambda x: x ** 2).sum(axis=1))
        return if_steps

    ####################################################################################################################
    """ fitting routines """

    def create_model(self, ModelType='Fabian2001', **parameters):
        """

        Parameters
        ----------
        type

        Returns
        -------

        """

        if ModelType == 'Fabian2001':
            self.model = RockPy.Packages.Magnetism.simulations.Fabian2001(preset='Fabian5a', **parameters)

    def fit_demag_data(self):
        pass

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

        def vds(self, vmin, vmax, **unused_params):  # todo move somwhere else
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

            y_dash = acqu_data[component] + (self.get_result('slope') * demag_data[component]) \
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

    class sigma(slope):
        pass

    class yint(slope):
        pass

    class xint(slope):
        pass

    class n(slope):
        pass

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
        dependencies = ('slope',)

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
        dependencies = ('slope', 'sigma')

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
            result = max_vd / sum_vd
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
        dependencies = ('q', 'n')

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
    def _format_vsm(ftype_data, **kwargs):
        """
        formatting routine for VSM type hysteresis measurements. Indirectly called by the measurement.from_file method.

        Parameters
        ----------
        ftype_data: RockPy.io
            data read by Magnetism.Ftypes.Vsm.vsm function.
            Contains:
              - data
              - header
              - segment_header


        Returns
        -------
            data: pandas.Dataframe
                columns = ['B','M']
            ftype_data: Ftypes object
                Ftypes object as read by Magnetism.Ftypes.Vsm.vsm


        """

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['Field (T)', 'Remanence (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            Dcd.log().error('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        segment_index = ftype_data.mtype.index('dcd')
        data = ftype_data.get_segment_data(segment_index).rename(columns={"Field (T)": "B", "Remanence (Am2)": "M"})

        return data

    @staticmethod
    def _format_vftb(ftype_data, sobj_name=None):  # todo implement VFTB
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

                plt.plot(-data.index, data['M'], '.', color=RockPy.colors[0], mfc='w', label='data')
                plt.plot(-x_new, y, color=RockPy.colors[0], label='fit')
                plt.plot(-result, 0, 'xk', label='B$_{cr}$')
                plt.axhline(0, color='k', zorder=0)

                plt.gca().text(0.05, 0.1, 'B$_{cr}$ = %.2f mT' % (abs(result) * 1000),
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
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.abs(result)
