import RockPy
import RockPy.core.utils
import pandas as pd
from RockPy.core.measurement import Measurement


class Parameter(Measurement):
    SIunit = None
    def __init__(self,
                 sobj,
                 fpath=None, ftype='generic',
                 series=None,
                 value=None,
                 column=None,
                 **options):


        if column is None:
            column = self.cls_mtype() + '[%s]' % self.SIunit

        if isinstance(value, str):
            value, unit = RockPy.core.utils.split_num_alph(value)

        if isinstance(value, (tuple, list)):
            if len(value) == 2:
                value, unit = value
            else:
                raise IndexError(
                        '%s can not be converted into readable format: try (value, unit)' % value)

        try:
            self.value = RockPy.core.utils.convert(value, unit, self.SIunit)
        except KeyError:
            self.log().error('Unit unknown')
            return

        mdata = pd.DataFrame(columns=[column], data=[[self.value]])

        super(Parameter, self).__init__(sobj=sobj,
                                        fpath=fpath, ftype=ftype,
                                        series=series, mdata=mdata,
                                        **options)

        # self.log().info(
        #         'creating %s: %.2f [%s] -> %.2e [%s]' % (self.__class__.get_subclass_name(),
        #                                                  value, unit, self.value, self.SIunit))

    def _format_generic(self):
        pass


class Mass(Parameter):
    """
    simple 1d measurement for mass
    """
    SIunit = 'kg'
    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 **options):
        super(Mass, self).__init__(sobj=sobj,
                                   fpath=fpath, ftype=ftype,
                                   series=series,
                                   value=value,
                                   unit='kg',
                                   **options)


class Length(Parameter):
    """
    simple 1d measurement for Length
    """
    SIunit = 'm'
    def __init__(self, sobj,
                 fpath=None, ftype='generic',
                 value=None,
                 series=None,
                 **options):
        super(Length, self).__init__(sobj=sobj,
                                     fpath=fpath, ftype=ftype,
                                     series=series,
                                     value=value,
                                     unit='m',
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

    m = RockPy.Packages.Generic.Parameter.Mass(sobj=s, value='2kg')
    m2 = RockPy.implemented_measurements['mass'](sobj=s, value='3kg')

    print(s.measurements[0].clsdata)
