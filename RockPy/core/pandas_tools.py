# here all functions, that manipulate panda Dataframes are  stored
import numpy as np


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


def gradient(df, ycol, xcol='Temperature (K)', append=False):
    """
    """
    df = df.reset_index().copy()

    dx = np.gradient(df[xcol])
    dy = np.gradient(df[ycol], df[xcol])

    df = df.set_index(xcol)  # todo fix issue where adding dx yields wrong results

    df['d({})/d({})'.format(ycol, xcol)] = dy
    return df if append else df['d({})/d({})'.format(ycol, xcol)]