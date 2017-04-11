import RockPy
import logging
import numpy as np
import pandas as pd
import os

from contextlib import contextmanager
from math import degrees, radians, atan2, asin, cos, sin, tan
from numba import jit

convert = pd.read_csv(os.path.join(RockPy.installation_directory, 'unit_conversion_table.csv'), index_col=0)

def XYZ2DIL(XYZ):
    """
    convert XYZ to dec, inc, length
    :param XYZ:
    :return:
    """
    DIL = []
    L = np.linalg.norm(XYZ)
    D = degrees(atan2(XYZ[1], XYZ[0]))  # calculate declination taking care of correct quadrants (atan2)
    if D < 0: D = D + 360.  # put declination between 0 and 360.
    if D > 360.: D = D - 360.
    DIL.append(D)  # append declination to Dir list
    I = degrees(asin(XYZ[2] / L))  # calculate inclination (converting to degrees)
    DIL.append(I)  # append inclination to Dir list
    DIL.append(L)  # append vector length to Dir list
    return DIL


def DIL2XYZ(DIL):
    """
    Convert a tuple of D,I,L components to a tuple of x,y,z.
    :param DIL:
    :return: (x, y, z)
    """
    (D, I, L) = DIL
    H = L * cos(radians(I))
    X = H * cos(radians(D))
    Y = H * sin(radians(D))
    Z = H * tan(radians(I))
    return (X, Y, Z)


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

@jit
def _to_tuple(oneormoreitems):
    """
    convert argument to tuple of elements

    Parameters
    ----------
        oneormoreitems: single number or string or list of numbers or strings

    Returns
    -------
        tuple of elements
    """
    return tuple(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else (
        oneormoreitems,)


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


def tuple2str(tup):
    """
    takes a tuple and converts it to text, if more than one element, brackets are put around it
    """
    if tup is None:
        return ''

    tup = _to_tuple(tup)

    if len(tup) == 1:
        return str(tup[0])
    else:
        return str(tup).replace('\'', ' ').replace(' ', '')


def split_num_alph(item):
    '''
    splits a string with numeric and str values into a float and a string

    Parameters
    ----------
    item: str
        The string that to be split

    Returns
    -------
        float, str
    '''
    # replace german decimal comma
    item.replace(',', '.')

    # cycle through all items in the string and stop at the first non numeric
    for i, v in enumerate(item):
        if not v in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.'):
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
           column_name_x:
           column_name_y:
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
