import RockPy
import RockPy.core.utils
import numpy as np
import pandas as pd
from RockPy.core.measurement import Measurement


class Parameter(Measurement):
    def __init__(self,
                 sobj,
                 fpath=None, ftype='generic',
                 series=None,
                 value=None,
                 Siunit=None,
                 column=None,
                 **options):

        super(Parameter, self).__init__(sobj=sobj,
                                        fpath=fpath, ftype=ftype,
                                        series=series,
                                        **options)

        unit = Siunit

        if column is None:
            column = self.mtype()+'[%s]'%unit

        if isinstance(value, str):
            value, unit = RockPy.core.utils.split_num_alph(value)

        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                value, unit = value
            else:
                raise IndexError(
                        '%s can not be converted into readable format: try (value, unit)' % value)

        try:
            unit_conversion = RockPy.core.utils.convert[unit][Siunit]
        except KeyError:
            self.log().error('Unit unknown')
            return

        if np.isnan(unit_conversion):
            self.log().warning(
                    'unit << %s >> most likely not %s-compatible' % (unit, self.__class__.get_subclass_name()))
            self.log().error('CAN NOT create Measurement')
            return

        # unit conversion into 'kg'
        self.value = value * unit_conversion
        self.unit = Siunit
        self.passed_unit = unit

        data = pd.DataFrame(columns=[column], data=[[self.value]])
        self.append_to_clsdata(data)

        self.log().info(
                'creating %s: %.2f [%s] -> %.2e [%s]' % (self.__class__.get_subclass_name(),
                                                         self.value / unit_conversion,
                                                         self.passed_unit, self.value, self.unit))

    def format_generic(self):
        pass


class Mass(Parameter):
    """
    simple 1d measurement for mass
    """

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 **options):
        super(Mass, self).__init__(sobj=sobj,
                                   fpath=fpath, ftype=ftype,
                                   series=series,
                                   value=value,
                                   Siunit='kg',
                                   **options)


class Length(Parameter):
    """
    simple 1d measurement for Length
    """

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 **options):
        super(Length, self).__init__(sobj=sobj,
                                     fpath=fpath, ftype=ftype,
                                     series=series,
                                     value=value,
                                     Siunit='m',
                                     **options)


class Diameter(Length):
    """
    simple 1d measurement for Length
    """

    pass


class Height(Length):
    """
    simple 1d measurement for Length
    """
    pass

if __name__ == '__main__':
    s = RockPy.Sample('test')
    s.add_measurement(mass = '13mg')
    s.add_measurement(mass = '12mg')

    m = RockPy.Packages.Generic.Measurement.Parameter.Mass(sobj=s, value='2kg')
    m2 = RockPy.implemented_measurements['mass'](sobj=s, value='3kg')

    print(s.measurements[0].clsdata)
