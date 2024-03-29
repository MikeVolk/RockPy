import numpy as np
from RockPy.core.utils import handle_shape_dtype
from RockPy.core.utils import convert
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

""" ROTATIONS """


def rx(angle):
    """Rotation matrix around X axis

    Args:
        angle (ndarray): Angle in radians to calculate the rotation matrix for

    Returns:
        ndarray: Rotationmatrix
    """
    RX = [[1, 0, 0],
          [0, np.cos(angle), -np.sin(angle)],
          [0, np.sin(angle), np.cos(angle)]]
    return RX


def ry(angle):
    """Rotation matrix around Y axis

    Args:
        angle (ndarray): Angle in radians to calculate the rotation matrix for

    Returns:
        ndarray: Rotationmatrix
    """
    RY = [[np.cos(angle), 0, np.sin(angle)],
          [0, 1, 0],
          [-np.sin(angle), 0, np.cos(angle)]]
    return RY


def rz(angle):
    """Rotation matrix around Z axis

    Args:
        angle (ndarray): Angle in radians to calculate the rotation matrix for

    Returns:
        ndarray: Rotationmatrix
    """
    RZ = [[np.cos(angle), -np.sin(angle), 0],
          [np.sin(angle), np.cos(angle), 0],
          [0, 0, 1]]
    return RZ


def rotmat(dec, inc):
    """Rotation matrix around Y axis

    Args:
        dec (ndarray): Angle in radians to calculate the rotation matrix for
        inc (ndarray): Angle in radians to calculate the rotation matrix for

    Returns:
        ndarray: Rotationmatrix
    """
    inc = np.radians(inc)
    dec = np.radians(dec)
    a = [[np.cos(inc) * np.cos(dec), -np.sin(dec), -np.sin(inc) * np.cos(dec)],
         [np.cos(inc) * np.sin(dec), np.cos(dec), -np.sin(inc) * np.sin(dec)],
         [np.sin(inc), 0, np.cos(inc)]]
    return np.array(a)


@handle_shape_dtype
def rotate_around_axis(xyz, *, axis_unit_vector, theta, axis_di=False, intype='xyz'):
    """Rotates a vector [x,y,z] or array of vectors around an arbitrary axis.

    Args:
        xyz (ndarray): data that shall get rotated
        axis_unit_vector (ndarray): axis around which the rotation is supposed
            to happen
        theta (float): angle of rotation in degrees
        axis_di (bool, False): if True the axis_unit_vector array contains
            declination and inclination values
        intype (str, 'xyz'): if 'dim' xyz array contains declination and
            inclination values. Note: intype is passed exclusively to yhr
            `handle_shape_dtype` decorator and needs to be a parameter.

    Returns:
        ndarray: Vectors that has been rotated by `theta` degrees around the
        specified axis ( `axis_di` ). if intype = 'dim' will return DIM values
    """

    if axis_di:
        axis_unit_vector = [axis_unit_vector[0], axis_unit_vector[1], 1]
        axis_unit_vector = convert_to_xyz(axis_unit_vector, M=False)
    # ensure the length of unit vector is 1
    axis_unit_vector = axis_unit_vector / np.linalg.norm(axis_unit_vector)

    ux, uy, uz = axis_unit_vector[0]

    theta = np.radians(theta)
    cost = np.cos(theta)
    sint = np.sin(theta)

    R = np.array([[cost + ux ** 2 * (1 - cost), ux * uy * (1 - cost) - uz * (sint), ux * uz * (1 - cost) + uy * (sint)],
                  [uy * ux * (1 - cost) + uz * sint, cost + uy ** 2 * (1 - cost), uy * uz * (1 - cost) - ux * sint],
                  [uz * ux * (1 - cost) - uy * sint, uz * uy * (1 - cost) + ux * sint, cost + uz ** 2 * (1 - cost)]])

    out = np.dot(R, xyz.T).T

    return out


@handle_shape_dtype
def rotate_arbitrary(xyz, *, alpha=0, beta=0, gamma=0, intype='xyz'):
    """
    Args:
        xyz (ndarray): data that shall get rotated
        alpha:
        beta:
        gamma:
        intype (str, 'xyz'): if 'dim' xyz array contains declination and
            inclination values. Note: intype is passed exclusively to yhr
            `handle_shape_dtype` decorator and needs to be a parameter.

    Returns:
        ndarray: data rotated around arbitrary axis if intype = 'dim' will
        return DIM values
    """

    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    r = np.dot(np.dot(rz(alpha), ry(beta)), rx(gamma))

    out = np.dot(r, xyz.T).T
    out = handle_near_zero(out)

    return out


@handle_shape_dtype
def rotate(xyz, *, axis='x', theta=0., intype='xyz'):
    """Rotates the vector 'xyz' by 'theta' degrees around 'x','y', or 'z' axis.

    Args:
        xyz (ndarray): data that shall get rotated
        axis (str): 'x', 'y', or 'z'
        theta (float, 0): angle of rotation in degrees
        intype (str, 'xyz'): if 'dim' xyz array contains declination and
            inclination values, where d = declination, i = inclination, and m =
            moment Note: intype is passed exclusively to yhr
            `handle_shape_dtype` decorator and needs to be a parameter.

    Returns:
        np.array: **out** -- The output array is in the same shape and type as
        the intype array
    """

    theta = np.radians(theta)

    if axis.lower() == 'x':
        out = np.dot(xyz, rx(theta))
    elif axis.lower() == 'y':
        out = np.dot(xyz, ry(theta))
    elif axis.lower() == 'z':
        out = np.dot(xyz, rz(theta))
    else:
        raise KeyError('only \'x\', \'y\', and \'z\' are valid options.')
    return out


@handle_shape_dtype(internal_dtype='dim')
def rotate_360_deg(xyz, theta, intype='xyz'):
    """draws a circle with angle theta around a point xyz

    circle: np.array

    Args:
        xyz:
        theta:
        intype:
    """

    circle = []
    # rotate around z axis
    for deg in np.arange(0, 360, 2):
        circle.append(rotate([0, 90 - theta, 1], axis='z', intype='dim', theta=deg)[0])
    circle = np.array(circle)

    # rotate that by 90-inc around 'y' axis (note: rotations are anticlockwise)
    circle = rotate(circle, axis='y', intype='dim', theta=-(90. - xyz[0][1]))

    # rotate that by dec around 'z' axis
    circle = rotate(circle, axis='z', intype='dim', theta=-xyz[0][0])

    return circle


""" CONVERSIONS """

"""
NOTE: conversion functions generally transform data from one coordinate system into a different one.
Therefor, the 'transform_output' value in the decorator has to be set to False, otherwise RockPy tries to 
bring the coordinates "back" into the original coordinate system, which will not work and give wrong values
"""


@handle_shape_dtype(transform_output=False)
def convert_to_xyz(dim, *, M=True):
    """Converts a numpy array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]])
    into an numpy array with [[d1,i1,m1], [d2,i2,m2]]. Reshape allows to pass an
    [[x1,x2],[y1,y2],[z1,z2]] array instead. Internally the data is handled in
    the (n,3) format.

    Args:
        dim (np.array, list): data either shape(n,3) of shape(3,n)
        M:

    Returns:
        np.array: **out** -- The output array is in the same shape and type as
        the intype array
    """
    dim = np.array(dim)

    D = dim[:, 0]
    I = dim[:, 1]

    if M:
        M = dim[:, 2]
    else:
        M = np.ones(len(D))

    M = 1 if M is None else M
    x = np.cos(np.radians(I)) * np.cos(np.radians(D)) * M
    y = np.cos(np.radians(I)) * np.sin(np.radians(D)) * M
    z = np.cos(np.radians(I)) * np.tan(np.radians(I)) * M

    out = np.array([x, y, z]).T
    return out


@handle_shape_dtype(transform_output=False)
def convert_to_dim(xyz):
    """Converts a numpy array of [d,i,m] values (i.e. [[d1,i1,m1], [d2,i2,m2]])
    into an numpy array with [[x1,y1,z1], [x2,y2,z2]] Reshape allows to pass an
    [[d1,d2],[i1,i2],[m1,m2]] array instead.

    Args:
        xyz:

    Returns:
        np.array: **out** -- The output array is in the same shape and type as
        the intype array
    """

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    M = np.linalg.norm([x, y, z], axis=0)  # calculate total moment for all rows
    D = np.degrees(np.arctan2(y, x)) % 360  # calculate D and map to 0-360 degree range
    I = np.degrees(np.arcsin(z / M))  # calculate I

    out = np.array([D, I, M]).T
    out = handle_near_zero(out)

    return out


@handle_shape_dtype(internal_dtype='dim', transform_output=False)
def convert_to_stereographic(xyz, intype='dim'):
    """Transforms an array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]])
    into an numpy array with [d,r,neg], where:

        d = declination, r = radius in equal area projection and neg = array of
        0,1 values where 0 = i > 0 and 1 = i < 0

    r is calculated according to Collinson, 1983 by .. math:

        R = 1 - ( R_0 * [ 1 - tan(frac{pi}{4} - frac{I}{2})]

    Args:
        xyz:
        intype:

    Returns:
        np.array: **out** -- The output array is in the same shape and type as
        the intype array

    See Also:
        :func:`convert_to_dim`, :func:`convert_to_xyz`,
        :func:`convert_to_equal_area`
    """

    # transformed by wrapper into DIM
    dim = xyz

    d = dim[:, 0]
    i = np.radians(dim[:, 1])
    neg = i < 0

    r = 1 - (1 - np.tan((np.pi / 4) - (abs(i) / 2)))

    out = np.array([d, r, neg]).T

    return out


@handle_shape_dtype(internal_dtype='xyz', transform_output=False)
def convert_to_equal_area(xyz, intype='xyz'):
    """
    Transforms an array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]])
    into an numpy array with [d,r,neg], where:

        d = declination, r = radius in equal area projection and neg = array of
        0,1 values where 0 = i > 0 and 1 = i < 0

    r is calculated according to Collinson, 1983 by

    .. math:
        r = 1 - ( R_0 * [ 1 - sqrt{1- sin(I)}])

    in the equal- area projection, where R0 is the radius of the projected circle.
    In polar projections it is usual to project the lower hemisphere only, and points on the upper hemisphere
    (negative inclinations) are transferred to the lower hemisphere (by changing the sign of I) and a different
    symbol used to distinguish them from positive inclinations.
    Conventionally, positive and negative inclinations are shown by full and open circles respectively.

    Args:
        xyz:
        intype:

    Returns:
        np.array: array in [theta, r, length] for plotting on polar coordinates
            -- The output array is in the same shape and type as the intype array

    See Also:
        :func:`convert_to_dim`, :func:`convert_to_xyz`,
        :func:`convert_to_stereographic`
    """

    # transformed by wrapper into DIM
    dim = convert_to_dim(xyz)

    d = dim[:, 0]
    i = dim[:, 1]
    neg = i < 0

    r0 = 1
    i = np.radians(np.abs(i))

    r = 1 - np.sqrt(1 - np.sin(i))
    # r = np.abs(xyz[:, 2]) # Tauxe
    out = np.array([d, r0 - r, neg]).T
    return out


@handle_shape_dtype(internal_dtype='dim', transform_output=False)
def convert_to_hvl(xyz, intype='xyz'):
    """Transforms an array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]])
    into an numpy array with [h,v,M], where:

        h = horizontal moment, v = vertical moment, and M = total moment

    h, v are calculated by .. math:

        H = M * cos(I) V = M * sin(I)

    Args:
        xyz:
        intype:

    Returns:
        np.array: The output array is in the same shape and type as the intype
        array

    See Also:
        :func:`convert_to_dim`, :func:`convert_to_xyz`,
        :func:`convert_to_stereographic`
    """

    # transformed by wrapper into DIM
    dim = xyz

    I = dim[:, 1]
    M = dim[:, 2]

    h = M * np.cos(np.radians(I))
    v = M * np.sin(np.radians(I))

    return np.array([h, v, M]).T


def handle_near_zero(d):
    """
    Args:
        d:
    """
    d[np.isclose(d, 0, atol=1e-15)] = 0
    return d


# if __name__ == '__main__':
#     from RockPy.tools.plotting import *
#
#     lst = [90, 90, 1]
#     lstlst = [lst, lst]
#     lstarr = [[lst[0], lst[0]], [lst[1], lst[1]], [lst[2], lst[2]]]
#
#     # res = convert_to_equal_area(lst)
#     for d in [lst, lstlst, lstarr]:
#         res = convert_to_equal_area(d)
#         # res = rotate_around_axis(d, axis_unit_vector=[135, 5], axis_di=True, theta=5)
#         print(np.shape(d), np.shape(res))
#         print(res)
#         print('-' * 30)
#     plot_equal(d, color='g', markersize=4, marker='o', ls='--', linecolor='k')
#
#     print(rotate_arbitrary(np.array([[30, 40, 50, 40], [0, 0, 10, 20], [0.1, 1, 1, 1]]), 0, 0, 0, dim=True))
def lin_regress(pdd, column_name_x, column_name_y, ypdd=None):
    """calculates a least squares linear regression for given x/y data

    Args:
        pdd (pandas.DataFrame): intype data
        column_name_x (str): xcolumn name
        column_name_y (str): ycolumn name
        ypdd (pandas.DataFrame): intype y-data. If not provided, it is asumed to
            be contained in pdd

    Returns:
        * **slope** (*float*)
        * **sigma** (*float*)
        * **y_intercept** (*float*)
        * **x_intercept** (*float*)
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


def detect_outlier(x, y, order, threshold):
    """fit data with polynomial :param x: :param y: :param order: :param
    threshold:

    Args:
        x:
        y:
        order:
        threshold:
    """

    p, residual, rank, singular_values, rcond = np.polyfit(x, y, order, full=True)
    rmse = np.sqrt(sum(residual) / len(x))  # root mean squared error
    p = np.poly1d(p)  # polynomial p(x)
    outliers = [i for i, v in enumerate(y) if v < p(x[i]) - threshold * rmse] + \
               [i for i, v in enumerate(y) if v > p(x[i]) + threshold * rmse]
    return outliers


def crossing_1d(x1, y1, x2, y2, lim=None, check=False, **kwargs):
    """Calculates the crossing of two datasets

    Args:
        x1:
        y1:
        x2:
        y2:
        lim:
        check (bool): (Default value = None) creates a diagnostic plot
        **kwargs:
    """
    f1 = interp1d(x1, y1, kind='slinear', bounds_error=False)
    f2 = interp1d(x2, y2, kind='slinear', bounds_error=False)
    
    if lim is None:
        xmin = min(np.append(x1, x2))
        xmax = max(np.append(x1, x2))
    else:
        (xmin, xmax) = lim

    xnew = np.arange(xmin, xmax, np.mean(np.diff(x1)) / 100)

    mnidx = np.nanargmin(abs(f1(xnew) - f2(xnew)))
    mn = xnew[mnidx]
    crossing = float(f1(mn))

    if check:
        plt.plot(xnew, f1(xnew), label='X1, Y1')
        plt.plot(xnew, f2(xnew), label='X2, Y2')
        plt.plot(xnew, abs(f1(xnew) - f2(xnew)), label='Y1-Y2')
        plt.plot(mn, crossing, 'rX')
        plt.legend()
        plt.show()
    return [mn, crossing]
