import os
from functools import partial, wraps

import json
import codecs
import configparser
import decorator
import importlib
import pkgutil
import RockPy
import numpy as np
import scipy as sp
import pandas as pd
import scipy.io as spio

from contextlib import contextmanager

import inspect
import logging

from RockPy import installation_directory, ureg

conversion_table = pd.read_csv(os.path.join(RockPy.installation_directory, 'unit_conversion_table.csv'), index_col=0)


def welcome_message():
    print('-' * 75)
    print(''.join(['|', 'This is RockPy'.center(73), '|']))
    print('-' * 75)
    print('Installation dir: %s' % installation_directory)
    print()
    print('IMPLEMENTED MEASUREMENT TYPES     : \tFTYPES')
    print('-' * 75)
    print('\n'.join(['\t{:<26}: \t{}'.format(m, ', '.join(obj.ftype_formatters().keys()))
                     for m, obj in sorted(RockPy.implemented_measurements.items())]))
    print()


def load_logging_conf(debug):
    """
    Loads the RockPy logging configuration
    Args:
        debug(bool): if debug or normal logging should be loaded #todo I think this does not work, fix 
    """
    if debug:
        logging.config.fileConfig(os.path.join(RockPy.installation_directory, 'logging_debug.conf'))
    else:
        logging.config.fileConfig(os.path.join(RockPy.installation_directory, 'logging.conf'))


def convert(value, unit, si_unit):
    """
    converts a value from a ``unit`` to a SIunit`` using `pint` package
    
    Args:
        value(float): value
        unit(str): input unit
        si_unit(str): desired output unit
    Returns:
        float: converted value
    """
    # print(value, unit, si_unit)
    converted = convert_units(value, in_unit=unit, out_unit=si_unit)
    if hasattr(value, 'magnitude'):
        value = value.magnitude
    RockPy.log.debug('converting %.3e [%s] -> %.3e [%s]' % (value, unit, converted, si_unit))
    return converted


def convert_units(values, in_unit, out_unit):
    """
    converts a value from a ``unit`` to a SIunit``

    Args:
        value(float): value
        in_unit(str): input unit
        out_unit(str): desired output unit

    Returns:
        float: converted value 
    """
    values = as_array(values)
    in_unit = values * to_quantity(in_unit)
    out_unit = in_unit.to(to_quantity(out_unit))
    return out_unit.magnitude


def to_quantity(unit):
    try:
        unit = ureg(unit)
    except AttributeError:
        pass
    return unit


@contextmanager
def ignored(*exceptions):
    """
    ignores certain exceptions

        Args:
    exceptions: errors
        the errors to be ignored

    """
    try:
        yield
    except exceptions:
        pass


def mtype_implemented(mtype):
    """
    Checks if given mtype is implemented in RockPy
    
    Args:
        mtype(str): mtype to check

    Returns:
        bool
    """
    return True if mtype in RockPy.implemented_measurements else False


''' ARRAY related '''


def as_array(df):
    """
    makes shure the output is an array, not Series/DataFrame
    
    Args:
        df(pandas.Series, pandas.DataFrame, ndarray): data to convert to array.

    Returns:
        ndarray: extracted/converted array
    """
    if isinstance(df, (pd.Series, pd.DataFrame)):
        return df.values
    elif hasattr(df, '__iter__'):
        return df
    else:  # todo
        return df


def tuple2list_of_tuples(item) -> list:
    """
    Takes a list of tuples or a tuple and returns a list of tuples

        Args:
       item(list, tuple):

    Returns:
       list: A list of tuples, if data is a tuple it converts it to a list of tuples 
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
    converts one or more items to a tuple of elements

        Args:
        oneormoreitems(int, float, list, array, tuple): single number or string or list of numbers or strings

    Returns:
        tuple: tuple of elements
    """
    return tuple(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else (
        oneormoreitems,)


def to_list(oneormoreitems):
    """
    conversion_table argument to tuple of elements

        Args:
        oneormoreitems: single number or string or list of numbers or strings

    Returns:
        tuple of elements
    """
    return list(oneormoreitems) if hasattr(oneormoreitems, '__iter__') and type(oneormoreitems) is not str else [
        oneormoreitems]


def str2tuple(s: str) -> tuple:
    """
    Extracts a tuple from a string, brackets ('[]()') are removed first

    e.g. "(HYS, COE)" -> ('hys','coe')
    e.g. "[HYS, COE]" -> ('hys','coe')

        Args:
        s(str): string to be tupeled

    Returns:
        tuple: tuple of strings

    See Also: tuple2str
    """
    s = s.translate(str.maketrans("", "", "(){}[]")).split(',')
    return tuple(s)


def tuple2str(tup):
    """
    takes a tuple and converts it to text, if more than one element, brackets are put around it

    Args:
        tup(tuple):

    Returns:
        str: string of the tuple, i.e. with surrounding brackets

    See Also: str2tuple
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

        Args:
        item(str): The string that to be split

    Returns:
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


def list_or_item(item):
    """
    Takes a list and returns a list if there is more than one element else it returns only that element.

        Args:
        item(list)

    Returns:
        list or first element of list
    """

    if np.shape(item)[0] == 1:
        return item[0]
    else:
        return item


@decorator.decorator
def correction(func, *args, **kwargs):
    """
    automatically adds the called correction_function to self._correct
    """
    self = args[0]

    set_get_attr(self, 'correction')

    if func.__name__ in self._correction:
        if not kwargs.pop('redo', False):
            self.log().warning('CORRECTION {} has already been applied'.format(func.__name__))
        return
    else:
        self.log().info('APPLYING correction {}, resetting results'.format(func.__name__))

    self.correction.append(func.__name__)
    func(*args, **kwargs)


def series_to_dict(series_tuple):
    """
    Converts a series tuple (stype, sval, sunit) to dictionary {stype: (sval, sunit)}

        Args:
    series_tuple: tuple len(3)

    Returns:

    dict

    """
    return {series_tuple[0]: (series_tuple[1], series_tuple[2])}


""" class and object related """


def extract_inheritors_from_cls(cls):
    """
    Method that gets all children and childrens-children ... from a class

    Args:
       cls(class):
    Returns:
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


def set_get_attr(obj, attr, value=None):
    """
    checks if attribute exists, if not, creates attribute with value None

        Args:
        obj (object):
        attr (str):
        value (str, int, float): default: None

    Returns:
        value(obj.attr)
    """
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    return getattr(obj, attr)


def get_default_args(func):
    signature = inspect.signature(func)

    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


""" Package related """


def import_submodules(package, recursive=True):
    """
    Import all submodules of a module, recursively, including subpackages

    Args:
        package(str, package): package name or actual module
        recursive(bool, True):
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


""" data related """


def handle_shape_dtype(func=None, internal_dtype='xyz', transform_output=True):
    """
    Decorator that transforms the input into an `xyz` array of (n,3) shape. If keyword 'dim' is provided,
    the data is assumed to be declination, inclination, and Moment and will first be converted to xyz
    values. Returns an array in its original form and coordinates.

    Args:
        func: wrapped function
        internal_dtype (str): tells the decorator what input data type is given
        transform_output (bool, True): transforms the data type back to the original input dtype.
            if False, calculated dtype will be returned
    Returns:
        ndarray: maintains shape and coordinates
    """
    if func is None:
        return partial(handle_shape_dtype, internal_dtype=internal_dtype, transform_output=transform_output)

    @wraps(func)
    def conversion(*args, **kwargs):

        # get defaults for the kwd args
        defaults = get_default_args(func)
        defaults.update(kwargs)
        kwargs = defaults

        if 'xyz' in kwargs:
            xyz = kwargs.pop('xyz')
        else:
            xyz = args[0]

        ## maintain vector shape part
        s = np.array(xyz).shape

        xyz = maintain_n3_shape(xyz)

        # handle input data
        # if the input data is dim it needs to be converted for functions, where internal dtype == 'xyz'
        if internal_dtype == 'dim':
            # if input data dtype == 'xyz' (i.e. input = 'xyz')
            if 'intype' in kwargs and kwargs['intype'] == 'xyz':
                from RockPy.tools.compute import convert_to_dim
                RockPy.log.debug(f'{func.__qualname__} uses \'dim\' for internal calculations: converting xyz -> dim')
                xyz = convert_to_dim(xyz)
        # if the internal dtype is xyz, input data in the format of 'dim' needs to be converted
        elif internal_dtype == 'xyz':
            # if input data dtype == 'xyz' (i.e. input = 'xyz')
            if 'intype' in kwargs and kwargs['intype'] == 'dim':
                from RockPy.tools.compute import convert_to_xyz
                RockPy.log.debug(f'{func.__qualname__} uses \'xyz\' for internal calculations: converting dim -> xyz')
                xyz = convert_to_xyz(xyz)

        # calculate function
        xyz = func(xyz, *args[1:], **kwargs)

        if transform_output:
            # return the same data type and shape as input
            # for internal dtype == dim, the data up to here is dim. Needs to be converted, if input was xyz.
            if internal_dtype == 'dim':
                # if input data dtype == 'xyz' (i.e. input = 'xyz')
                if 'intype' in kwargs and kwargs['intype'] == 'xyz':
                    from RockPy.tools.compute import convert_to_xyz
                    xyz = convert_to_xyz(xyz)

            # if the internal dtype is xyz, input data in the format of 'dim' needs to be converted
            elif internal_dtype == 'xyz':
                # if input data dtype == 'xyz' (i.e. input = 'xyz')
                if 'intype' in kwargs and kwargs['intype'] == 'dim':
                    from RockPy.tools.compute import convert_to_dim
                    xyz = convert_to_dim(xyz)

        return xyz

    return conversion


@handle_shape_dtype
def handle_shape(xyz):
    return xyz


def maintain_n3_shape(xyz):
    """
    Takes vector of (3,), (n,3) and (3,n) shape and transforms it into (n,3) shape used for ALL compute calculations.

    Args:
        xyz (ndarray): data to be returned

    Returns:
        ndarray: in the shape of (n,3)

    Raises:
        ValueError:
            - if not at least one dimension is len 3
            - if the number of elements is inconsistent (e.g. [[1, 2, 3], [1, 2], [1, 2]])
    """
    ## maintain vector shape part
    s = np.array(xyz).shape

    if not any(i == 3 for i in s):
        raise ValueError('At least one dimension needs to be length 3')
    # for [x,y,z] or [d,i,m]
    if s == (3,):
        if len(set(np.shape(elem) for elem in xyz)) != 1:
            raise ValueError('Number of elements ix xyz is inconsistent')
        xyz = np.array(xyz).reshape((1, 3))

    # for array like [[x1,x2,... ],[y1,y2,...],[z1,z2,...]],
    elif s[0] == 3 and s[1] != 3:
        xyz = np.array(xyz).T
    elif s[1] == 3 and s[0] != 3:
        xyz = np.array(xyz)
    else:
        RockPy.log.warning('Input cannot be interpreted, due to ambiguous shape. '
                           'Input could be [[x1,x2,x3],[y1,y2,y3],[z1,z2,z3]] or [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].'
                           'Returning original shape')
        xyz = np.array(xyz)
    return xyz

def loadmat(filename):
    '''
    FROM: https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            # elif isinstance(elem, np.ndarray):
            #     d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = sp.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


_MagIC_codes = None

def MagIC_codes():
    if RockPy.core.utils._MagIC_codes is None:
        data = json.load(
            codecs.open(os.path.join(RockPy.installation_directory, 'MagIC_method_codes.json'), 'r', 'utf-8-sig'))
        _MagIC_codes = {n: {i['code']: i['definition'] if 'definition' in i else None for i in data[n]['codes']} for n
                        in data.keys()}
    return _MagIC_codes

def CreateConfigFile():
    config = configparser.ConfigParser()
    for mtype, cls in sorted(RockPy.implemented_measurements.items()):
        for result in cls._result_classes():
            config['#'.join([mtype, result])] = {}
            print(result)
            # print(cls.res_signature()[result]['signature'])
            # config[mtype][] = 'test'#cls.res_signature()[result]
    #         #         if result == 'b_anc':
    #         #             print(cls.res_signature()[result])
    #         #         if not cls.res_signature()[result]['indirect']:
    #         #             standard_method = '_'.join([result, cls.result_recipe()[result]]).replace('_DEFAULT', '')
    #         #             for param, value in cls.calc_signature()[standard_method].items():
    #         #                 line = ', '.join([mtype, result, cls.result_recipe()[result].lower(), param, str(value), '\n'])
    #         #                 f.write(line)
    #
    with open(os.path.join(RockPy.installation_directory, 'configfile.ini'), 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    CreateConfigFile()
    # import RockPy.packages.magnetism.measurements
    # s = RockPy.Sample('test')
    # m = s.add_measurement(fpath='/Users/mike/github/RockPy/RockPy/tests/test_data/VSM/hys_vsm.001',
    #                       mtype='hys',ftype='vsm')
    # for i in m._result_classes():
    #     print(i)
    # for i in RockPy.packages.magnetism.measurements.Hysteresis._result_classes():
    #     print(i)