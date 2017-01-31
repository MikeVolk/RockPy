import pandas as pd

import RockPy.core.utils
from RockPy.core.measurement import Measurement
import RockPy


class Parameter(Measurement):
    def format_generic(self):
        pass


class Mass(Parameter):
    """
    simple 1d measurement for mass
    """

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 mass=None,
                 series=None,
                 **options):

        super(Mass, self).__init__(sobj=sobj,
                                   fpath=fpath, ftype=ftype,
                                   series=series,
                                   **options)
        unit = 'kg'

        if isinstance(mass, str):
            mass, unit = RockPy.core.utils.split_num_alph(mass)

        if isinstance(mass, (tuple, list)):
            if len(mass) == 2:
                mass, unit = mass
            else:
                raise IndexError(
                    '%s can not be converted into readable format: try (mass, unit) or \'mass unit\'' % mass)

        # unit conversion into 'kg'
        if unit:
            mass *= RockPy.core.utils.convert[unit]['kg']


        self.unit = unit if unit else 'kg'
        data = pd.DataFrame(columns=['mass'], data=[[mass]])
        self.append_to_clsdata(data)
        self.log().info(
            'creating mass: %.2e [%s] -> %.2e [%s]' % (mass * RockPy.core.utils.convert['kg'][unit], unit, mass, 'kg'))


# class Length(Parameter):
#     """
#     simple 1d measurement for Length
#     """
#
#     def __init__(self, sobj,
#                  fpath=None, ftype='generic',
#                  value=None, unit='m',
#                  direction=None,
#                  series=None,
#                  **options):
#         super(Length, self).__init__(sobj=sobj,
#                                      fpath=fpath, ftype=ftype,
#                                      series=series,
#                                      **options)
#         if not value:
#             return
#
#         self.ftype = ftype
#         self.direction = direction
#
#         # length_conversion = convert2(unit, 'm', 'length')
#
#         if not length_conversion:
#             self.log().warning('unit << %s >> most likely not %s-compatible' % (unit, self.__class__.get_subclass_name()))
#             self.log().error('CAN NOT create Measurement')
#             return
#
#         self._data = {'data': RockPy3.Data(column_names=[self.mtype, 'time', 'std_dev'])}
#         self._data['data'][self.mtype] = value * length_conversion
#         self._data['data']['time'] = time
#         self._data['data']['std_dev'] = std
#
#     def format_generic(self):
#         pass
#
#
# class Diameter(Length):
#     """
#     simple 1d measurement for Length
#     """
#
#     pass
#
#
# class Height(Length):
#     """
#     simple 1d measurement for Length
#     """
#     pass

if __name__ == '__main__':
    s = RockPy.Sample('test', mass='1.3827ng')
    print(s.measurements[0]._clsdata)
