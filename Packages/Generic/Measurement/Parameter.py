import pandas as pd
from RockPy.core.measurement import Measurement
import RockPy.core.utils


class Parameter(Measurement):
    pass

    def format_generic(self):
        pass


class Mass(Parameter):
    """
    simple 1d measurement for mass
    """

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 mass=None,
                 std=None, time=None,
                 series=None,
                 **options):

        super(Mass, self).__init__(sobj=sobj,
                                   fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)

        if isinstance(mass, str):
            mass, unit = RockPy.core.utils.split_num_alph(mass)

        if isinstance(mass, (tuple, list)):
            if len(mass) == 2:
                mass, unit = mass
            else:
                raise IndexError(
                    '%s can not be converted into readable format: try (mass, unit) or \'mass unit\'' % mass)
        # todo add mass conversion

        self.unit = unit if unit else 'kg'
        self._data = pd.DataFrame(columns=['mass'], data=[[mass]])

        self.log.info('creating mass: %f, %s' % (mass[0], self.unit))

    def format_test(self):
        pass

    def format_testerich(self):
        pass

if __name__ == '__main__':
    test = Mass('test', mass='23.7 mg', series=('test', 2, 'au'))
    print(test.data)
