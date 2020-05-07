import logging
import os
from shutil import copy2

import pandas as pd
import RockPy.core.utils
from RockPy.core.file_io import ImportHelper
from copy import deepcopy
from io import StringIO, BytesIO
import pint
from RockPy import ureg
class Ftype(object):
    """ Delivers core functionality to all classes inheriting from it.

    Args:
        in_units (dict(:obj:`pint.ureg`)): internal_units from used in the data file
        out_units (dict(:obj:`pint.ureg`)): internal_units used to export the data
        units (dict(:obj:`pint.ureg`)): internal_units used internally (should be SI internal_units)
        fid (int): internal id of the instance
        _raw_data (list(str)): raw, imported data from the file. Note: has not been converted to SI internal_units.
        snames (tuple(str)): tuple of all names in the datafile
        dfile (str): full path to the file on the HD
        dialect (str, optional): can change the behavior of the `Ftype.read_file` method
        header (:obj:`pandas.DataFrame`, optional): header data generated from the data file. May not exist for certain ftypes
        importhelper (:obj:`RockPy.core.file_io.ImportHelper`, optional): Object that helps importing files using
            the `RockPy` naming convention. Only created if `create_minfo` is True
        file_infos (dict, optional): Information on the file. Only created if `create_minfo` = True
        imported_files (dict): Secondary storage of a deepcopy of the `obj.data`, for possible recovery.

    Notes:
        The internal_units that the data is stored in are SI internal_units. They can be converted using `RockPy.core.utils.convert`.

    TODO:
        - use `os.scandir` instead of `os.listdir`
    """
    imported_files = {}
    # std internal_units for import
    in_units = {}
    # std internal_units for export
    out_units = {}
    # desired internal internal_units
    units = {}

    def __init__(self, dfile,
                 snames=None,
                 dialect=None,
                 reload=False, create_minfo=True,
                 mdata=None,
                 **kwargs):
        """ Constructor of the basic file type instance

        Creates the instance of the class `RockPy.core.ftype.Ftype`.

        Args:
            dfile (str): path to the file
            snames (str, list): name(s) of the sample(s)
            dialect (str, optional): used if a specific file was modified from version to version.
                Specifies the particular version of the file.
            reload (bool, optional): defaults to False.  Specifies if the file should be loaded from the HD again.
                If False, the file will not be loaded for each sample if there are more than one.
                If True, data will be reloaded (e.g. when using ipython Notebooks)
            mdata (:obj:`pd.DataFrame`, optional):  data to be returned using `obj.data`.
                Needs to be formatted the same way as data read by `obj.read_file`.
                This should passed from pseudo-abstract methods such as cls.from_XXX to the cls constructor.
            create_minfo (bool, optional): defaults to True. If True an `RockPy` will try to create an instance
                of `RockPy.core.file_io.ImportHelper`, which is stored.
            **kwargs: additional infos (e.g. header) passed to the constructor from cls.from_XXX methods
        """

        # set file ID
        self.fid = id(self)

        # initialize the _raw_data attribute
        self._raw_data = None
        # create sample name tuple
        self.snames = [str(i) for i in RockPy.core.utils.to_tuple(snames)] if snames else None

        self.dfile = dfile

        dio = False
        ## catch ERROR for StringIO and ByteIO
        if isinstance(dfile, (StringIO, BytesIO)):
            self.log().debug('impporting Byte/String data')
            dfile = self.fid
            dio = True

        self.dialect = dialect

        # only add header if it hasnt been defined already
        ### the header should now be passed to the constructor
        if getattr(self, 'header', None) is None:
            # set attribute in case header is provided in **kwargs, default to None
            self.header = kwargs.pop('header', None)

        if create_minfo:
            try:
                self.import_helper = ImportHelper.from_file(self.dfile)
                self.file_infos = self.import_helper.return_file_infos()
            except:
                self.__class__.log().warning('Cant read file infos automatically. RockPy file naming scheme required')
                self.import_helper = None
                self.file_infos = None

        if mdata is not None:
            self.__class__.imported_files[dfile] = mdata

        elif dfile not in self.__class__.imported_files or reload:
            if not dio:
                self.log().info('IMPORTING << %s , %s >> file: << ... %s >>' % (self.snames,
                                                                            type(self).__name__, dfile[-20:]))
            else:
                self.log().info('IMPORTING << %s , %s >> from Bytes' % (self.snames, type(self).__name__))

            self.__class__.imported_files[dfile] = self.read_file()

        else:
            self.log().info(
                "LOADING previously imported file << {} , {} >> file: << {} >>\n\t>>> "
                "USE reload option if you want to read files from HD".format(self.snames, type(self).__name__, dfile[-40:]))

        self.data = self.__class__.imported_files[dfile]
        self.to_si_units()

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.subclass_name().lower())

    @staticmethod
    def split_tab(line):

        return line.split('\t')

    @classmethod
    def subclass_name(cls):

        return cls.__name__

    @classmethod
    def inheritors(cls):

        return RockPy.core.utils.extract_inheritors_from_cls(cls)

    def _get_comment_line_indices(self):
        """ Checks for commented lines in the file and returns the indices of those lines. Comments are '#'

        Returns:
            list: list of integer indices
        """
        with open(self.dfile) as f:
            idx = [i for i, d in enumerate(f.readlines()) if d.startswith('#')]
        if idx:
            self.log().info('FILE contains {} comments. Lines {} were not read'.format(len(idx), idx))
        return idx

    def _has_specimen(self, specimen):
        """ Checks if specimen is contained in file

        Args:
            specimen (str): name of the specimen

        Returns:
             bool: `True` if file contains `specimen` otherwise `False`

        """
        if specimen not in self.data['specimen'].values:
            Ftype.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Ftype.log().error('wrong sample name?')
            Ftype.log().error('These samples exist: %s' % set(self.data['specimens']))
            return False
        else:
            return True

    def to_si_units(self):
        """ converts each numeric column in self.data to SI internal_units TODO: write to out_units
        """
        for col in self.data.columns:
            if col in self.in_units:
                if not col in self.units:
                    self.log().warning(
                        'Unit of data column << {} >> has no internal unit equivalent. The input unit is  << {:P} >>.'.format(
                            col, self.in_units[col]))
                    continue
                if self.in_units[col] == self.units[col]:
                    self.log().debug(
                        'Data column << {} >> already in internal unit << {:P} >>.'.format(
                            col, self.in_units[col]))
                    continue

                in_unit = self.in_units[col]
                unit = self.units[col]

                # convert to internal unit
                try:

                    conv = (1 * in_unit).to(unit).magnitude
                    self.log().debug(f'converting to SI units {in_unit} -> {conv * unit}')
                    self.data.loc[:, col] *= conv

                except pint.DimensionalityError:
                    self.log().debug(f'Pint conversion to SI units FAILED {in_unit} -> {unit}')
                    if in_unit == ureg('gauss') and unit == ureg('tesla'):
                        conv  = 1e-4
                    if in_unit ==  ureg('tesla') and unit == ureg('gauss'):
                        conv = 1e4
                    self.data.loc[:, col] *= conv
                    self.log().info(f'manual conversion to SI units {in_unit} -> {conv} {unit}')


    def read_file(self):
        """ Method for actual import of the file.

        This method needs to be written for each filetype. It is called in the __init__ constructor.

        See Also:
            The read_file methods of classes inheriting from :obj:`Ftype`
        """
        pass

    def rename_file_using_RockPy_convention(self, backup=True):
        """ Renames the file using the NEW RockPy naming convention.

        The function tries to create the new style RockPy file name from the internal file infos.
        By default it creates a copy of the original file, that is commented by using '#' in from of the name.
        """

        old_filename = os.path.basename(self.dfile)
        new_filename = list(self.import_helper.new_filenames)[0]

        if backup:
            self.log().info('Creating a copy of file << %s >>' % old_filename)
            copy2(self.dfile, self.dfile.replace(old_filename, '#' + old_filename))
        self.log().info('Renaming file << %s >> to << %s >>' % (old_filename, new_filename))
        os.rename(self.dfile, self.dfile.replace(old_filename, new_filename))

    @property
    def raw_data(self):
        """ raw data of the file.

        If _raw_data is None (initialized in constructor, `read_raw_data` is called and assigned to `_raw_data`.
        """
        if self._raw_data is None:
            self._raw_data = self.read_raw_data(dfile=self.dfile)
        return self._raw_data

    @staticmethod
    def read_raw_data(dfile):
        """ reads file from HD.

        Reads the file from HD using:
            >>> with open(dfile, 'r', encoding="ascii", errors="surrogateescape") as f:

        Args:
            dfile (str): full path to the file

        Returns:
            list(str): the data read with `encoding = 'asci'`, and `errors = 'surrogateescape'`
        """
        with open(dfile, 'r', encoding="ascii", errors="surrogateescape") as f:
            raw_data = f.readlines()
        return raw_data

    def copy(self):
        """
        returns a deepcopy of itself, with new `id`

        Returns:
            :obj:`RockPy.core.ftype.Ftype`:
        """
        out =  deepcopy(self)
        out.fid = id(out)
        return out

    def unit_label(self, quantity):
        return '{:~P}'.format(self.units[quantity].units)

### related functions
def is_implemented(ftype):
    """ Checks if ftype has been implemented.

    Args:
        ftype (str): name of the ftype

    Returns:
        bool: Returns `True` if implemented, `False` if not

    Raises:
        `log.error` No exception
    """
    if ftype in RockPy.implemented_ftypes:
        return True
    else:
        RockPy.log.error(f'Ftype << {ftype} >> is not implemented. Check RockPy.implemented_ftypes for which ftypes are.')
        return False
