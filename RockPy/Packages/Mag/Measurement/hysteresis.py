import RockPy
from RockPy.core.measurement import Measurement
from RockPy.core.result import Result
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


class Hysteresis(Measurement):


    @property
    def data(self):
        out = super().data
        return out.set_index('B')
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
        expected_columns = ['Field (T)', 'Moment (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            Hysteresis.log().debug('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        data = ftype_data.data.rename(columns={"Field (T)": "B", "Moment (Am2)": "M"})
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
        uf_fields = np.arange(-1*self.max_field, self.max_field + self.fieldspacing, self.fieldspacing)
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

    def get_polarity_switch(self, window = 1):
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
        diffs = diffs.fillna(method = 'bfill') #filling missing values at beginning
        diffs = diffs.fillna(method = 'ffill') #filling missing values at end

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
        return (np.where(signchange!=0)[0]).astype(int)

    @property
    def downfield(self):
        '''

        Returns
        -------
            pandas.DataFrame with only downfield data. Window size for selecting the polarity change is 5
        '''
        #todo how to change window size?
        idx = self.get_polarity_switch_index(5)
        if len(idx) > 1:
            return self.data.iloc[int(idx[0]):int(idx[1])].dropna()
        else:
            return self.data.iloc[0:int(idx[1])].dropna()

    @property
    def upfield(self):
        '''

        Returns
        -------
            pandas.DataFrame with only upfield data. Window size for selecting the polarity change is 5
        '''
        #todo how to change window size?
        idx = self.get_polarity_switch_index(5)
        return self.data.iloc[int(idx[-1])-1:].dropna()

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

    def get_reversible(self): # todo implement reversible part
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

    class bc(Result):
        default_recipe = 'linear'

        def recipe_linear(self, npoints=4, check=False, **unused_params):
            """
            Calculates the coercivity using a linear interpolation between the points crossing the x axis for upfield and down field slope.

            Parameters
            ----------
                field_limit: float
                    default_recipe: 0, 0mT
                    the maximum/ minimum fields used for the linear regression

            Note
            ----
                Uses scipy.linregress for calculation
            """
            # initialize result
            result = []

            m = self.mobj

            # get magnetization limits for a calculation using the n points closest to 0 fro each direction
            df_moment = sorted(abs(m.downfield['M'].values))[npoints - 1]
            uf_moment = sorted(abs(m.upfield['M'].values))[npoints - 1]

            # filter data for fields higher than field_limit
            down_f = m.downfield[m.downfield['M'].abs() <= df_moment]
            up_f = m.upfield[m.upfield['M'].abs() <= uf_moment]

            # calculate bc for both measurement directions
            for i, dir in enumerate([down_f, up_f]):

                # calculate the linear regression
                res = stats.linregress(dir['M'],dir.index)
                result.append(res[1])

                # check plot
                if check:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(dir.index, dir['M'])
                    x = dir.index
                    y_new = slope * x + intercept
                    l, = plt.plot(x, dir['M'], '.', label='data')
                    plt.plot(x, y_new, '--', color=l.get_color(), label='fit')

            # check plot
            if check:
                plt.plot(result, [0, 0], 'ko', mfc='w', label='Bc (branch)')
                plt.plot([-np.nanmean(np.abs(result)), np.nanmean(np.abs(result))], [0, 0], 'xk', label='mean Bc')
                plt.grid()
                plt.xlabel('Field')
                plt.ylabel('Moment')
                plt.title('Bc - check')
                plt.legend()
                plt.text(0.8,0.1, '$B_c = $ %.1f mT'%(np.nanmean(np.abs(result))*1000),
                         transform=plt.gca().transAxes,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

                plt.tight_layout()
                plt.show()

            result = np.abs(result)
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nanmean(result)

        def recipe_nonlinear(self, npoints=8, check=False, **unused_params):
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
                Uses scipy Univariate spline for interpolation
            """
            # retireve measurement instance
            m = self.mobj

            # initialize result
            result = []

            # the field_limit has to be set higher than the lowest field
            # if not the field_limit will be chosen to be 2 points for uf and df separately
            if npoints < 2:
                self.log().warning('NPOINTS INCOMPATIBLE minimum 2 required' % (npoints))
                self.log().warning('\t\t setting NPOINTS - << 2 >> ')
                npoints = 2
                self.params['npoints'] = npoints

            # get magnetization limits for a calculation using the n points closest to 0 fro each direction
            df_moment = sorted(abs(m.downfield['M'].values))[npoints - 1]
            uf_moment = sorted(abs(m.upfield['M'].values))[npoints - 1]

            # filter data for fields higher than field_limit
            down_f = m.downfield[m.downfield['M'].abs() <= df_moment]
            up_f = m.upfield[m.upfield['M'].abs() <= uf_moment]

            for i, dir in enumerate([down_f, up_f]):
                spl = UnivariateSpline(dir['M'].values, dir.index)
                bc = spl(0)
                result.append(bc)

                if check:
                    spl = UnivariateSpline(dir.index, dir['M'].values)
                    x_new = np.linspace(min(dir.index), max(dir.index), 100)
                    l, = plt.plot(x_new, spl(x_new), '--', label='fit')
                    plt.plot(dir.index, dir['M'], '.', label='data', color=l.get_color())

            if check:
                plt.plot(result, [0,0], 'ko', mfc='w', label='Bc(branch)')
                plt.plot([-np.nanmean(np.abs(result)), np.nanmean(np.abs(result))], [0, 0], 'xk', label='mean Bc')
                plt.grid()
                plt.xlabel('Field')
                plt.ylabel('Moment')
                plt.title('Bc [%s]- check'%'nonlinear')
                plt.text(0.8,0.1, '$B_c = $ %.1f mT'%(np.nanmean(np.abs(result))*1000),
                         transform=plt.gca().transAxes,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

                plt.legend()
                plt.show()

            result = np.abs(result)
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nanmean(result)

    ####################################################################################################################
    """ MRS """

    class mrs(Result):
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
                plt.text(0.8,0.1, '$M_{rs} = $ %.1f '%(np.nanmean(np.abs(result))),
                         transform=plt.gca().transAxes,
                         bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})

                plt.tight_layout()
                plt.show()

            result = np.abs(result)
            self.mobj.sobj.results.loc[self.mobj.mid, self.name] = np.nanmean(result)

    ####################################################################################################################
    """ MS """

    class ms(Result):
        def get_df_uf_plus_minus(self, saturation_percent, ommit_last_n):
            """
            Filters the data :code:`down_field`, :code:`up_field` to be larger than the saturation_field, filters the last :code:`ommit_last_n` and splits into pos and negative components
            """
            # transform from percent value
            saturation_percent /= 100

            # filter ommitted points
            if ommit_last_n >0:
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

        # @calculate
        # def calculate_ms_APP2SAT(self, saturation_percent=75., ommit_last_n=0, check=False, **non_method_parameters):
        #     """
        #     Calculates the high field susceptibility using approach to saturation
        #     :return:
        #     """
        #
        #     ms, slope, alpha = self.calc_approach2sat(saturation_percent=saturation_percent,
        #                                               ommit_last_n=ommit_last_n)
        #
        #     self.results['ms'] = [[[np.mean(ms), np.std(ms)]]]
        #     self.results['hf_sus'] = [[[np.mean(slope), np.std(slope)]]]
        #     self.results = self.results.append_columns(column_names=['alpha'],
        #                                                data=[[[np.mean(slope), np.std(slope)]]])
        #
        def recipe_simple(self, saturation_percent=75., ommit_last_n=0, check=False, **unused_params):
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
                calculates the slope using SciPy.linregress for each branch at positive and negative fields. Giving four values for the slope. The result is the mean for all four values, and the error is the standard deviation

            """

            # initialize result
            hf_sus_result = []
            ms_result = []

            if saturation_percent >= 100:
                self.log().warning('SATURATION > 100%! setting to default_recipe value (75%)')
                saturation_percent = 75.0

            df_plus, df_minus, uf_plus, uf_minus= self.get_df_uf_plus_minus(saturation_percent=saturation_percent,
                                                                            ommit_last_n=ommit_last_n)

            # calculate for each branch for positive and negative fields
            for i, dir in enumerate([df_plus, df_minus, uf_plus, uf_minus]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(dir.index, dir['M'])
                hf_sus_result.append(abs(slope))
                ms_result.append(abs(intercept))

                # check plot
                if check:
                    d0 = self.get_df_uf_plus_minus(0,0)
                    x = np.linspace(0, self.mobj.max_field)
                    y_new = slope * x + abs(intercept)
                    l, = plt.plot(abs(d0[i].index), d0[i]['M'].abs(), ':', lw=1, label='data')
                    plt.plot(x, y_new, '--', color=l.get_color())

            # check plot
            if check:
                # plt.plot([0,0,0,0], np.abs(ms_result), 'ko', label='Ms (branch)', mfc='none', mew=0.5)
                plt.errorbar([0], np.mean(np.abs(ms_result)), yerr=2*np.std(np.abs(ms_result)),
                             color='k', marker='.', label='mean Ms', zorder=100, )
                plt.axvline(self.mobj.max_field*saturation_percent/100, ls='--', color='grey')
                plt.xlabel('Field')
                plt.ylabel('Moment')
                plt.xlim([-self.mobj.max_field*0.01,self.mobj.max_field])

                plt.grid()
                plt.show()

            self.mobj.sobj.results.loc[self.mobj.mid, 'hf_sus'] = np.nanmean(np.abs(hf_sus_result))
            self.mobj.sobj.results.loc[self.mobj.mid, 'ms'] = np.nanmean(np.abs(ms_result))

    class hf_sus(ms):
        dependencies = ['ms']


if __name__ == '__main__':

    s = RockPy.Sample('test')
    m = s.add_measurement(mtype='hys', ftype='vsm',
                          fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/hys_vsm.001')

    # import matplotlib.pyplot as plt
    # plt.plot(m.upfield['M'], color='r')
    # # plt.gca().twinx().plot(m.get_polarity_switch().values)
    # plt.plot(m.downfield['M'], color='b')
    # plt.show()

    # print(m.downfield.shape, m.upfield.shape)

    # for r in m._results:
    #     print(r, m._results[r]._recipes())
    # m.calc_all(check=True)
    # print(m.result_bc(npoints=6, check=True))
    # print(m.result_bc(npoints=15, check=True))
    # print(m.result_bc(recipe='nonlinear', npoints=10, check=True))
    # print(m.result_ms(ommit_last_n=4, check=True, saturation_percent=90))
    print(m.result_hf_sus(check=True))
    # print(m.results)

