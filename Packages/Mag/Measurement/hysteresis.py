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
        return data, ftype_data

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

    def field_polarity_switch_idx(self):
        '''

        Returns
        -------

        '''
        a = self.data['B']
        asign = np.sign(a)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

        return signchange

    @property
    def downfield(self):
        raise NotImplementedError

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

