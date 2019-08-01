import os
import decorator

import RockPy
import numpy
import numpy as np
import pandas as pd

from contextlib import contextmanager

from Cython.Includes import numpy

conversion_table = pd.read_csv(os.path.join(RockPy.installation_directory, 'unit_conversion_table.csv'), index_col=0)


def convert(value, unit, si_unit):
    """
    converts a value from a ``unit`` to a SIunit``
    
    Parameters
    ----------
    value
    unit
    si_unit

    Returns
    -------
        float
        
    Notes
    -----
        the conversion table is stored in RockPy.installation_directory as 'unit_conversion_table.csv'
    """
    RockPy.log.debug(
        'converting %.3e [%s] -> %.3e [%s]' % (value, unit, value * conversion_table[unit][si_unit], si_unit))
    return value * conversion_table[unit][si_unit]


@contextmanager
def ignored(*exceptions):
    """
    ignores certain exceptions

    Parameters
    ----------
    exceptions: errors
        the errors to be ignored

    """
    try:
        yield
    except exceptions:
        pass


def mtype_implemented(mtype):
    """
    Function to check if mtype is implemented
    
    Parameters
    ----------
    mtype: str
        mtype to check

    Returns
    -------
        bool
    """
    return True if mtype in RockPy.implemented_measurements else False


def tuple2list_of_tuples(item) -> list:
    """
    Takes a list of tuples or a tuple and returns a list of tuples

    Parameters
    ----------
       item: list, tuple

    Returns
    -------
       list
          Returns a list of tuples, if data is a tuple it converts it to a list of tuples
          if data == a list of tuples will just return data
    """

    # check if item is a list -> each item in item has to be converted to a tuple
    if isinstance(item, list):
        for i, elem in enumerate(item):
            if not type(elem) == tuple:
                item[i] = (elem,)

    if not isinstance(item, (list, tuple)):
        item = tuple([item])

    if isinstance(item, tuple):
        item = [item, ]

    return item


def to_tuple(oneormoreitems):
    """
    conversion_table argument to tuple of elements

    Parameters
    ----------
        oneormoreitems: single number or string or list of numbers or strings

    Returns
    -------
        tuple of elements
    """
    return tuple(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else (
        oneormoreitems,)


def to_list(oneormoreitems):
    """
    conversion_table argument to tuple of elements

    Parameters
    ----------
        oneormoreitems: single number or string or list of numbers or strings

    Returns
    -------
        tuple of elements
    """
    return list(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else [
        oneormoreitems]


def extract_tuple(s: str) -> tuple:
    """
    Extracts a tuple from a string, brackets ('[]()') are removed first

    e.g. "(HYS, COE)" -> ('hys','coe')
    e.g. "[HYS, COE]" -> ('hys','coe')

    Parameters
    ----------
    s str
        string to be tupeled

    Returns
    -------
        tuple
    """
    s = s.translate(str.maketrans("", "", "(){}[]")).split(',')
    return tuple(s)
#
# if __name__ == '__main__':
#     print(extract_tuple('(a,b)'))

def tuple2str(tup):
    """
    takes a tuple and converts it to text, if more than one element, brackets are put around it
    """
    if tup is None:
        return ''

    tup = to_tuple(tup)

    if len(tup) == 1:
        return str(tup[0])
    else:
        return str(tup).replace('\'', ' ').replace(' ', '')


def split_num_alph(item):
    """
    splits a string with numeric and str values into a float and a string

    Parameters
    ----------
    item: str
        The string that to be split

    Returns
    -------
        float, str
    """
    # replace german decimal comma
    item.replace(',', '.')

    idx = None
    # cycle through all items in the string and stop at the first non numeric
    for i, v in enumerate(item):
        if v not in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.'):
            break
        else:
            idx = i

    if not idx == len(item) - 1:
        return float(item[:idx + 1]), item[idx + 1:].strip()
    else:
        return float(item), None


def lin_regress(pdd, column_name_x, column_name_y, ypdd=None):
    """
        calculates a least squares linear regression for given x/y data

        Parameters
        ----------
           pdd: pandas.DataFrame
            input data
           column_name_x: str
            xcolumn name
           column_name_y: str
            ycolumn name
           ypdd: pandas.DataFrame
            input y-data. If not provided, it is asumed to be contained in pdd
        Returns
        -------
            slope
            sigma
            y_intercept
            x_intercept
        """
    x = pdd[column_name_x].values

    if ypdd is not None:
        y = ypdd[column_name_y].values
    else:
        y = pdd[column_name_y].values

    if len(x) < 2 or len(y) < 2:
        return None

    """ calculate averages """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    """ calculate differences """
    x_diff = x - x_mean
    y_diff = y - y_mean

    """ square differences """
    x_diff_sq = x_diff ** 2
    y_diff_sq = y_diff ** 2

    """ sum squared differences """
    x_sum_diff_sq = np.sum(x_diff_sq)
    y_sum_diff_sq = np.sum(y_diff_sq)

    mixed_sum = np.sum(x_diff * y_diff)

    """ calculate slopes """
    n = len(x)

    slope = np.sqrt(y_sum_diff_sq / x_sum_diff_sq) * np.sign(mixed_sum)

    if n <= 2:  # stdev not valid for two points
        sigma = np.nan
    else:
        sigma = np.sqrt((2 * y_sum_diff_sq - 2 * slope * mixed_sum) / ((n - 2) * x_sum_diff_sq))

    y_intercept = y_mean - (slope * x_mean)
    x_intercept = - y_intercept / slope

    return slope, sigma, y_intercept, x_intercept


def set_get_attr(obj, attr, value=None):
    """
    checks if attribute exists, if not, creates attribute with value None

    Parameters
    ----------
        obj: object
        attr: str
        value: (str, int, float)
            default: None

    Returns
    -------
        value(obj.attr)
    """
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    return getattr(obj, attr)


@decorator.decorator
def correction(func, *args, **kwargs):
    """
    automatically adds the called correction_function to self._correct
    """
    self = args[0]

    set_get_attr(self, 'correction')

    if func.__name__ in self._correction:
        self.log().warning('CORRECTION {} has already been applied'.format(func.__name__))
        return
    else:
        self.log().info('APPLYING correction {}, resetting results'.format(func.__name__))

    self.correction.append(func.__name__)
    func(*args, **kwargs)


def series_to_dict(series_tuple):
    """
    Converts a series tuple (stype, sval, sunit) to dictionary {stype: (sval, sunit)}

    Parameters
    ----------
    series_tuple: tuple len(3)

    Returns
    -------

    dict

    """
    return {series_tuple[0]:(series_tuple[1], series_tuple[2])}


def list_or_item(item):
    """
    Takes a list and returns a list if there is more than one element else it returns only that element.

    Parameters
    ----------
    item: List

    Returns
    -------
        list or first element of list
    """

    if np.shape(item)[0] == 1:
        return item[0]
    else:
        return item

def rotmat(dec, inc):
    inc = np.radians(inc)
    dec = np.radians(dec)
    a = [[np.cos(inc)*np.cos(dec), -np.sin(dec), -np.sin(inc)*np.cos(dec)],
         [np.cos(inc)*np.sin(dec), np.cos(dec), -np.sin(inc)*np.sin(dec)],
         [np.sin(inc), 0 , np.cos(inc)]]
    return a

def extract_inheritors_from_cls(cls):
        """
        Method that gets all children and childrens-children ... from a class

        Returns
        -------
           list
        """
        subclasses = set()
        work = [cls]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses:
                    subclasses.add(child)
                    work.append(child)
        return subclasses