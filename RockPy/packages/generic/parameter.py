import RockPy
import RockPy.core.utils
import pandas as pd
from RockPy.core.measurement import Measurement
from RockPy import ureg


class Parameter(Measurement):
    SIunit = None

    def __init__(self,
                 sobj,
                 fpath=None, ftype='generic',
                 series=None,
                 value=None, unit=None,
                 column=None,
                 **options):

        # if the value is a string (i.e. '20 mg' ect.) create the object using from_string method
        if isinstance(value, str):
            value, unit = RockPy.core.utils.split_num_alph(value)

        if column is None:
            column = self.cls_mtype() + ' [{:~P}]'.format(self.SIunit.units)

        unit = RockPy.core.utils.to_quantity(unit)
        if unit != self.SIunit:
            try:
                self.value = RockPy.core.utils.convert(value, unit, self.SIunit)
            except KeyError:
                self.log().error('Unit unknown')
                raise KeyError
        else:
            self.value = float(value)
        mdata = pd.DataFrame(columns=[column], data=[[self.value]])

        super().__init__(sobj=sobj,
                         fpath=fpath, ftype=ftype,
                         series=series, mdata=mdata,
                         **options)


    @classmethod
    def from_string(cls, sobj, string,
                    fpath=None, ftype='generic',
                    series=None,
                    column=None):

        cls.log().debug(f'Creating << {cls.cls_mtype()} >> from string')
        value, unit = RockPy.core.utils.split_num_alph(string)
        return cls(sobj=sobj, fpath=fpath, ftype=ftype, series=series, column=column, value=value, unit=unit)

    def _format_generic(self):
        pass

    def __repr__(self):
        if self.is_mean:
            add = 'mean_'
        else:
            add = ''
        return '<<RockPy.{}.{}{}{} {} ({}) at {}>>'.format(self.sobj.name, add, self.mtype,
                                                    '[' + ';'.join(['{},{}({})'.format(i[0], i[1], i[2]) for i in
                                                                    self.get_series()]) + ']' if self.has_series() else '',
                                                    self.data.iloc[0,0], self.SIunit.units,
                                                    hex(self.mid))
class Mass(Parameter):
    """
    simple 1d measurement for mass
    """
    SIunit = ureg('kg')

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 unit='kg',
                 **options):
        if unit is None:
            unit = 'kg'
        super().__init__(sobj=sobj,
                         fpath=fpath, ftype=ftype,
                         series=series,
                         value=value,
                         unit=unit,
                         **options)


class Length(Parameter):
    """
    simple 1d measurement for Length
    """
    SIunit = ureg('m')

    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 unit='m',
                 **options):

        if unit is None:
            unit = 'm'

        super().__init__(sobj=sobj,
                         fpath=fpath, ftype=ftype,
                         series=series,
                         value=value,
                         unit=unit,
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
    RockPy.debug_mode()
    s = RockPy.Sample('test')
    # m = s.add_measurement(mass='13mg')
    # m2 = s.add_measurement(mass='12mg')

    m = RockPy.packages.generic.parameter.Mass(sobj=s, value='2kg')
    print(m)
    # m2 = RockPy.implemented_measurements['mass'](sobj=s, value='3kg')

    for m in s:
        print(m.data)
