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

        mass, unit = RockPy.core.utils.split_num_alph(mass)
        # todo add mass conversion

        self.unit = unit if unit else 'kg'
        self._data = pd.DataFrame(columns=['mass', 'unit'], data=[[mass, self.unit]])


if __name__ == '__main__':
    test = Mass('test', mass='23.7 mg')
    print(test.data)
