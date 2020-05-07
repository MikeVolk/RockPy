# here all functions, that manipulate panda Dataframes are  stored
import numpy as np
import matplotlib.pyplot as plt
import RockPy
import pandas as pd
from RockPy.tools import compute
from RockPy.tools.compute import rotate, convert_to_xyz


def dim2xyz(df, colD='D', colI='I', colM='M', colX='x', colY='y', colZ='z'):
    """
    adds x,y,z columns to pandas dataframe calculated from D,I,(M) columns

    Args:
        df (pandas.DataFrame): data including columns of D, I and optionally M values
        colD (str, 'D'): name of column with declination input data
        colI (str, 'I'): name of column with inclination input data
        colM (str, 'M'): name of column with moment data (will be set to 1 if None)
        colX (str, 'X'): name of column for x data (will be created or overwritten)
        colY (str, 'Y'): name of column for y data (will be created or overwritten)
        colZ (str, 'Z'): name of column for z data (will be created or overwritten)
    """
    df = df.copy()
    dim = df[[colD, colI, colM]]
    df_xyz = pd.DataFrame(columns=[colX, colY, colZ], data=convert_to_xyz(dim))
    df = pd.concat([df, df_xyz], axis=1)
    return df


def xyz2dim(df, colX='x', colY='y', colZ='z', colD='D', colI='I', colM='M'):
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
    df = df.copy()
    xyz = df[[colX, colY, colZ]]
    df_dim = pd.DataFrame(columns=[colX, colY, colZ], data=convert_to_xyz(xyz))
    df = pd.concat([df, df_dim], axis=1)
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


def gradient(*args, **kwargs):
    print('depricated, please change to derivative')
    return derivative(*args, **kwargs)


def derivative(df, ycol, xcol='index', n=1, append=False, rolling=False, edge_order=1, norm=False, **kwargs):
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
    df_copy = df_copy.reset_index()
    df_copy = df_copy[[xcol, ycol]]
    # calculate the rolling mean before differentiation
    if rolling:
        kwargs.setdefault('center', True)

        if isinstance(rolling, int):
            kwargs.setdefault('window', rolling)

        df_copy = df_copy.rolling(**kwargs).mean()

    # reset index, so that the index col can be accessed
    x = df_copy[xcol].astype(float)
    y = df_copy[ycol].astype(float)
    # print(x, y)
    dy = np.gradient(y, x, edge_order=edge_order)

    if n == 2:
        dy = np.gradient(dy, x, edge_order=edge_order)

    if norm:
        dy /= np.nanmax(abs(dy))

    col_name = 'd{}({})/d({}){}'.format(n, ycol, xcol, n).replace('d1', 'd').replace(')1', ')')

    out = pd.DataFrame(data=np.array([x, dy]).T, columns=[xcol, col_name])
    out = out.set_index(xcol)

    if append:
        out = pd.concat([df, out])
    return out


def detect_outlier(df, column, threshold=3, order=4):
    """
    Detects Outliers by first fitting a polynomial p(x) of order <order. to the data. Then calculates the root mean
    square error from the residuals. The data is then compared to the fit ± the threshold * RMSe.
    All points that are outside this boundary, are considered an outlier.

    Parameters
    ----------
    df: pandas.Dataframe
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
    x, y = (df.index, df[column])
    return compute.detect_outlier(x, y, order, threshold)


def remove_outliers(df, column, threshold=3, order=4, **kwargs):
    """
    Removes outliers from pandas.DataFrame using detect_outliers.

    Parameters
    ----------
    df: pandas.DataFrame
    column: str
        column to detect outliers in
    threshold: int
        default: 3
        multiples of the RMSerror

    Returns
    -------
    DataFrame without outliers
    """
    outliers = detect_outlier(df, column, threshold, order)
    RockPy.log.info(
        "removing %i outliers that are exceed the %.2f standard deviation threshold" % (len(outliers), threshold))

    df = df.drop(df.index[outliers])

    return df


def regularize_data(df, order=2, grid_spacing=2, ommit_n_points=0, check=False, **parameter):
    """
        Parameters
    ----------

       method: str
          method with which the data is fitted between grid points.

          first:
              data is fitted using a first order polinomial :math:`M(B) = a_1 + a2*B`
          second:
              data is fitted using a second order polinomial :math:`M(B) = a_1 + a2*T +a3*B^2`

       parameter: dict
          Keyword arguments passed through

    See Also
    --------
       get_grid
    """

    d = df
    d = d.sort_index()

    dmax = max(d.index)
    dmin = min(d.index)

    grid = np.arange(dmin, dmax + grid_spacing, grid_spacing)

    if check:
        uncorrected_data = d.copy()

    # initialize DataFrame for gridded data
    interp_data = pd.DataFrame(columns=df.columns)

    if ommit_n_points > 0:
        d = d.iloc[ommit_n_points:-ommit_n_points]

    # cycle through gridpoints
    for i, T in enumerate(grid):
        for col in df.columns:
            # set T to T column
            interp_data.loc[i, df.index.name] = T

            # indices of points within the grid points
            if i == 0:
                idx = [j for j, v in enumerate(d.index) if v <= grid[i]]
            elif i == len(grid) - 1:
                idx = [j for j, v in enumerate(d.index) if grid[i] <= v]
            else:
                idx = [j for j, v in enumerate(d.index) if grid[i - 1] <= v <= grid[i + 1]]

            if len(idx) > 1:  # if no points between gridpoints -> no interpolation
                data = d.iloc[idx]

                # make fit object
                fit = np.polyfit(data.index, data[col].values.astype(float), order)

                # calculate Moment at grid point
                dfit = np.poly1d(fit)(T)

                interp_data.loc[i, col] = dfit

            # set dtype to float -> calculations dont work -> pandas sets object
            interp_data[col] = interp_data[col].astype(np.float)
    interp_data = interp_data.set_index(df.index.name)

    if check:
        plt.plot(uncorrected_data, marker='.', mfc='none', color='k')
        plt.plot(interp_data, '-', color='r')

    return interp_data


def correct_dec_inc(df, dip, strike, newI='I_', newD='D_', colD='D', colI='I'):
    """
    Function that corrects the Dec and Inc values of a DataFrame by dip/strike

    1. rotates aroud y-axis by -dip (i.e. counter clockwise)
    2. rotates around z axis by -strike (i.e. counter clockwise)

    Parameters
    ----------
    df: pd.DataFrame
    dip: float
        dip of the 'core', 'plate'...
    strike: float
        strike of the 'core', 'plate' ...
    newI: str
        default 'I_'
        name of the corrected inclination column
    newD: str
        default 'D_'
        name of the corrected inclination column
    colI: str
        default 'I'
        name of the uncorrected inclination column
    colD: str
        default 'D'
        name of the uncorrected inclination column

    Returns
    -------

    """
    df = df.copy()
    DI = df[[colD, colI]]
    DI = dim2xyz(DI, colI=colI, colD=colD, colM=None)

    xyz = DI[['x', 'y', 'z']]

    xyz = rotate(xyz, axis='y', theta=-dip)
    xyz = rotate(xyz, axis='z', theta=-strike)

    corrected = xyz2dim(pd.DataFrame(columns=['x', 'y', 'z'], data=xyz, index=DI.index),
                        colI=newI, colD=newD)

    df[newI] = corrected[newI]
    df[newD] = corrected[newD]
    return df


def get_values_in_both(a, b, key='level', return_sorted=True):  # todo TEST
    '''
    Looks through pd.DataFrame(a)[key] and pd.DataFrame(b)[key] to find values in both

    Parameters
    ----------
    a: pd.DataFrame
        first DataFrame
    b: pd.DataFrame
        second DataFrame
    key: str

    return_sorted: bool, optional
        default: True
        if True the values are returned sorted
        if False values are returned as is
    Returns
    -------
        sorted(list) of items
    '''

    if key == 'index':
        aval = a.index
        bval = b.index
    else:
        aval = a[key].values
        bval = b[key].values

    equal_vals = np.array(list(set(aval) & set(bval)))

    if return_sorted:
        return np.sort(equal_vals)
    else:
        return equal_vals


def interpolate(df, levels, retain_levels=True, **kwargs):
    """
    Interpolates a dataframe to new index values.

    Parameters
    ----------
    df
    levels
    retain_levels
    kwargs: dict
        passed on to pandas.DataFrame.interpolate
    Returns
    -------
        pandas.DataFrame
    a copy of the original DataFrame with the interpolated values
    """
    df = df.copy()

    if retain_levels:
        levels_not_included = df.index[np.in1d(df.index, levels)].values
        if any(levels_not_included):
            levels = np.concatenate([levels, levels_not_included])
            levels = np.sort(levels)
    df = df.reindex(levels)
    df = df.interpolate(**kwargs)
    return df


def remove_duplicate_index(df, method='duplicated', **kwargs):
    if method == 'duplicated':
        return df[~df.index.duplicated(keep=kwargs.pop('keep', 'first'))]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t = pd.DataFrame(index=np.linspace(0, 10, 10), data=np.linspace(0, 10, 10) ** 2)

    plt.plot(t)
    plt.plot(interpolate(t, np.linspace(0, 10), method='akima'))

    plt.show()
