import RockPy
from RockPy.core.measurement import Measurement
import numpy as np

class Hysteresis(Measurement):

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

        if self.data['B'].ix[0] > 0.9 * self.data['B'].max():
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

        print(self.data['B'].max(), self.data['B'].min())

    """ CALCULATIONS """

    def get_irreversible(self, correct_symmetry=True):
        """
        Calculates the irreversible hysteretic components :math:`M_{ih}` from the data.

        .. math::

           M_{ih} = (M^+(H) + M^-(H)) / 2

        where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop

        Returns
        -------
           Mih: RockPyData

        """

        uf = self.data['up_field']  # .interpolate(field_data)
        field_data = uf[
            'field'].v  # sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))

        df = self.data['down_field'].interpolate(field_data)

        M_ih = deepcopy(uf)
        M_ih['mag'] = (df['mag'].v + uf['mag'].v) / 2

        if correct_symmetry:
            M_ih_pos = M_ih.filter(M_ih['field'].v >= 0).interpolate(field_data)
            M_ih_neg = M_ih.filter(M_ih['field'].v <= 0).interpolate(field_data)
            mean_data = np.nanmean(np.c_[M_ih_pos['mag'].v, -M_ih_neg['mag'].v][::-1], axis=1)
            M_ih['mag'] = list(-mean_data).extend(list(mean_data))

        return M_ih.filter(~np.isnan(M_ih['mag'].v))

    def get_reversible(self):
        """
        Calculates the reversible hysteretic components :math:`M_{rh}` from the data.

        .. math::

           M_{ih} = (M^+(H) - M^-(H)) / 2

        where :math:`M^+(H)` and :math:`M^-(H)` are the upper and lower branches of the hysteresis loop

        Returns
        -------
           Mrh: RockPyData

        """
        # field_data = sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))
        # uf = self.data['up_field'].interpolate(field_data)

        uf = self.data['up_field']  # .interpolate(field_data)
        field_data = uf[
            'field'].v  # sorted(list(set(self.data['down_field']['field'].v) | set(self.data['up_field']['field'].v)))

        df = self.data['down_field'].interpolate(field_data)
        M_rh = deepcopy(uf)
        M_rh['mag'] = (df['mag'].v - uf['mag'].v) / 2
        return M_rh.filter(~np.isnan(M_rh['mag'].v))


if __name__ == '__main__':
    s = RockPy.Sample('test')
    m = s.add_measurement(mtype='hys', ftype='vsm',
                          fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/hys_vsm.002')

    m.field_polarity_switch_idx()
