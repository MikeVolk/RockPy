import RockPy
from RockPy.core.measurement import Measurement
import numpy as np

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
        ftype_data: RockPy.Ftype
            data read by Mag.Ftype.Vsm.vsm function.
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
            ftype_data: Ftype object
                Ftype object as read by Mag.Ftype.Vsm.vsm


        '''

        # expected column names for typical VSM hysteresis experiments
        expected_columns = ['Field (T)', 'Moment (Am2)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            Hysteresis.log().debug('ftype_data has more than the expected columns: %s' % list(ftype_data.data.columns))

        data = ftype_data.data.rename(columns={"Field (T)": "B", "Moment (Am2)": "M"})
        return data

    ####################################################################################################################
    ''' BRANCHES '''

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
            default: 1 - no running mean
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
            default: 1 - no running mean
            Size of the moving window. This is the number of observations used for calculating the statistic.

        Returns
        -------
            np.array of indices
        '''

        asign = self.get_polarity_switch()

        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

        return (np.where(signchange!=0)[0] + window/2).astype(int)

    @property
    def downfield(self):
        '''

        Returns
        -------

        '''
        sign = self.get_polarity_switch(1)
        return self.data[sign < 0]
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    s = RockPy.Sample('test')
    m = s.add_measurement(mtype='hys', ftype='vsm',
                          fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/hys_vsm.001')

    print(m.clsdata)
