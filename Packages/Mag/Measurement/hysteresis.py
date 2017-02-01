import RockPy
from RockPy.core.measurement import Measurement


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
        expected_columns = ['Field (T)', 'Moment (Am�)']

        if not all(i in expected_columns for i in ftype_data.data.columns):
            print(ftype_data.data.columns)

        data = ftype_data.data[expected_columns]
        data.columns = ['B', 'M']
        return data, ftype_data


if __name__ == '__main__':
    s = RockPy.Sample('test')
    m = s.add_measurement(mtype='hys', ftype='vsm',
                          fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/hys_vsm.001')
    m.data.to_csv('/Users/mike/Desktop/data.csv')
