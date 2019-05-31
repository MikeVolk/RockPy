# here all functions, that manipulate panda Dataframes are  stored
import numpy as np
import pandas as pd


def DIM2XYZ(df, colD='D', colI='I', colM=None, colX='x', colY='y', colZ='z'):
    """
    adds x,y,z columns to pandas dataframe calculated from D,I,(M) columns

    Parameters
    ----------
    df: pandas dataframe
        data including columns of D, I and optionally M values
    colD: str
        name of column with declination input data
    colI: str
        name of column with inclination input data
    colM: str
        name of column with moment data (will be set to 1 if None)
    colX: str
        name of column for x data (will be created or overwritten)
    colY: str
        name of column for y data (will be created or overwritten)
    colZ: str
        name of column for z data (will be created or overwritten)

    Returns
    -------

    """
    M = 1 if colM is None else df[colM]

    df[colX] = np.cos(np.radians(df[colI])) * np.cos(np.radians(df[colD])) * M
    df[colY] = np.cos(np.radians(df[colI])) * np.sin(np.radians(df[colD])) * M
    df[colZ] = np.cos(np.radians(df[colI])) * np.tan(np.radians(df[colI])) * M
    return df


def XYZ2DIM(df, colX='x', colY='y', colZ='z', colD='D', colI='I', colM=None):
    """
    adds D,I,(M) columns to pandas dataframe calculated from x,y,z columns

    Parameters
    ----------
    df: pandas dataframe
        data including columns of x, y, z values
    colX: str
        name of column for x input data
    colY: str
        name of column for y input data
    colZ: str
        name of column for z input data
    colD: str
        name of column with declination data (will be created or overwritten)
    colI: str
        name of column with inclination data (will be created or overwritten)
    colM: str
        name of column with moment data (will not be written if None)

    Returns
    -------

    """

    M = np.linalg.norm([df[colX], df[colY], df[colZ]], axis=0)  # calculate total moment for all rows
    df[colD] = np.degrees(np.arctan2(df[colY], df[colX])) % 360  # calculate D and map to 0-360 degree range
    df[colI] = np.degrees(np.arcsin(df[colZ] / M))  # calculate I
    if colM is not None:
        df[colM] = M  # set M
    return df


def heat(df, tcol='index'):
    """
    returns only values where temperature in Tcol is increasing

    Parameters
    ----------
    df: pandas dataframe
        data including columns of D, I and optionally M values
    tcol: str
        name of column with temperature input data, may be the index

    Returns
    -------
    pandas dataframe with only the heating data
    """

    if tcol == 'index':
        df = df[np.gradient(df.index) > 0]
    else:
        df = df[np.gradient(df[tcol]) > 0]
    return df


def cool(df, tcol='index'):
    """
    returns only values where temperature in Tcol is decreasing

    Parameters
    ----------
    df: pandas dataframe
        data including columns of D, I and optionally M values
    tcol: str
        name of column with temperature input data, may be the index

    Returns
    -------
    pandas dataframe with only the heating data
    """

    if tcol == 'index':
        df = df[np.gradient(df.index) < 0]
    else:
        df = df[np.gradient(df[tcol]) < 0]
    return df


def gradient(df, ycol, xcol='index', n=1, append=False, rolling=False, edge_order=1, norm=False, **kwargs):
    """
    Calculates the derivative of the pandas dataframe. The xcolumn and ycolumn have to be specified.
    Rolling adds a rolling mean BEFORE differentiation is done. The kwargs can be used to change the rolling.

    Parameters
    ----------
    df: pandas.DataFrame
        data to be differentiated

    xcol: str
        column name of the x values. Can be index then index.name is used. Default: 'index'

    ycol: str
        column name of the y column

    # append: bool #todo implement
    #     - if True: the column is appended to the original dataframe
    #     - if False: the column is returned individually
    #     Default: False

    rolling: bool, int
        Uses a rolling mean before differentiation. Default: False
        if integer is given, the int is used as ''window'' parameter in DataFrame.rolling

    n: ({1, 2}, optional)
        Degree of differentiation. Default:1

    edgeorder: ({1, 2}, optional)
        Gradient is calculated using N-th order accurate differences at the boundaries. Default: 1

    norm: bool
        returns data normalized to the maximum

    kwargs:
        passed to rolling

    Returns
    -------
        Pandas dataframe
    """
    # use index column if specified, if index is unnemaded, use 'index'
    if xcol == 'index' and df.index.name is not None:
        xcol = df.index.name

    df_copy = df.copy()

    # calculate the rolling mean before differentiation
    if rolling:
        kwargs.setdefault('center', True)

        if isinstance(rolling, int):
            kwargs.setdefault('window', rolling)

        df_copy = df_copy.rolling(**kwargs).mean()

    # reset index, so that the index col can be accessed
    df_copy = df_copy.reset_index()
    x = df_copy[xcol]
    y = df_copy[ycol]
    dy = np.gradient(y, x, edge_order=edge_order)

    if n == 2:
        dy = np.gradient(dy, x, edge_order=edge_order)

    if norm:
        dy /= max(abs(dy))

    col_name = 'd{}({})/d({}){}'.format(n, ycol, xcol, n).replace('d1', 'd').replace(')1', ')')

    out = pd.DataFrame(data=np.array([x, dy]).T, columns=[xcol, col_name])
    out = out.set_index(xcol)

    return out

    def normalize(df):
        pass


from scipy import stats


def detect_outlier(pdd, column, threshold=3, order=4):
    """
    Detects Outliers by first fitting a polynomial p(x) of order <order. to the data. Then calculates the root mean
    square error from the residuals. The data is then compared to the fit ± the threshold * RMSe.
    All points that are outside this boundary, are considered an outlier.

    Parameters
    ----------
    pdd: pandas.Dataframe
    column: str
        column to detect outliers in
    threshold: int
        default: 3
        multiples of the RMSerror
    order: int
        default: 4
        order of the polynomial

    Returns
    -------
    list
        list of indices
    """
    x, y = (pdd.index, pdd[column])
    # fit data with polynomial
    z, res, _, _, _ = np.polyfit(x, y, order, full=True)

    rmse = np.sqrt(sum(res) / len(x)) # root mean squared error
    p = np.poly1d(z) # polynomial p(x)

    outliers = [i for i, v in enumerate(pdd[column]) if v < p(x[i]) - threshold * rmse] + \
               [i for i, v in enumerate(pdd[column]) if v > p(x[i]) + threshold * rmse]

    return outliers

def remove_outliers(pdd, column, threshold=3, **kwargs):
    """
    Removes outliers from pandas.Dataframe using detect_outliers.

    Parameters
    ----------
    pdd: pandas.Dataframe
    column: str
        column to detect outliers in
    threshold: int
        default: 3
        multiples of the RMSerror

    Returns
    -------
    Dataframe without outliers
    """

    order = kwargs.pop('order', 4)
    outliers = detect_outlier(pdd, column, threshold, order)

    pdd = pdd.drop(pdd.index[outliers])

    return pdd