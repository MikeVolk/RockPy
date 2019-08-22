import logging
import os
from shutil import copy2

import pandas as pd
import RockPy.core.utils
from RockPy.core.file_io import ImportHelper
from copy import deepcopy

class Ftype(object):
    imported_files = {}

    # std units for import
    in_units = {}
    # std units for export
    out_units = {}
    # internal units
    units = {}

    def has_specimen(self, specimen):
        if specimen not in self.data['specimen'].values:
            Ftype.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Ftype.log().error('wrong sample name?')
            Ftype.log().error('These samples exist: %s' % set(self.data['specimens']))
            return False
        else:
            return True

    def to_si_units(self):
        for col in self.data.columns:
            if col in self.in_units:
                if not col in self.units:
                    self.log().warning('Unit of data column << {} >> has no internal equivalent. The input unit is  << {:P} >>.'.format(col, self.in_units[col]))
                    continue
                inunit = self.in_units[col]
                outunit = self.units[col]

                self.data.loc[:,col] *= (1*inunit).to(outunit).magnitude


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

    def get_comment_line_indices(self):
        """
        Checks for commented lines in the file and returns the indices of those lines. Comments are '#'

        Returns
        -------
            list
                list of integer indices
        """
        with open(self.dfile) as f:
            idx = [i for i, d in enumerate(f.readlines()) if d.startswith('#')]
        if idx:
            self.log().info('FILE contains {} comments. Lines {} were not read'.format(len(idx), idx))
        return idx

    def __init__(self, dfile, snames=None, dialect=None,
                 reload=False, mdata=None, create_minfo=True,
                 **kwargs):
        """
        Constructor of the basic file type instance

        Parameters
        ----------
        dfile: str
            path to the file
        snames: list
            name(s) of the sample(s)
        dialect: str
            used if a specific file was modified from version to version. Specifies the particular version of the file.
        reload: bool
            default: False
            Specifies if the file should be loaded from the HD again.
            If False, the file will not be loaded for each sample if there are more than one.
            If True, data will be reloaded (e.g. when using ipython Notebooks)
        mdata: pd.DataFrame
            formatted the same way as self.data. This should passed from cls.from_XXX methods to the cls constructor
        kwargs
            additional infos (e.g. header) passed to the constructor from cls.from_XXX methods

        """

        # set file ID
        self.fid = id(self)

        self._raw_data = None
        # create sample name tuple
        self.snames = [str(i) for i in RockPy.core.utils.to_tuple(snames)] if snames else None
        self.dfile = dfile
        self.dialect = dialect
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
            self.data = mdata
            self.__class__.imported_files[dfile] = self.data.copy()

        elif dfile not in self.__class__.imported_files or reload:
            self.log().info('IMPORTING << %s , %s >> file: << ... %s >>' % (self.snames,
                                                                        type(self).__name__, dfile[-20:]))
            self.data = self.read_file()
            self.__class__.imported_files[dfile] = self.data.copy()
        else:
            self.log().info(
                "LOADING previously imported file << {} , {} >> file: << {} >>\n\t>>> "
                "USE reload option if you want to read files from HD".format(self.snames,
                                                                             type(self).__name__, dfile))

            self.data = self.__class__.imported_files[dfile]

        self.to_si_units()

    def read_file(self):
        """
        Method for actual import of the file. Needs to be written for each individual filetype.
        """
        pass

    def rename_file_using_RockPy_convention(self, backup=True):
        """
        Renames the file using the NEW RockPy naming convention.
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
        if self._raw_data is None:
            self._raw_data = self.read_raw_data(dfile=self.dfile)
        return self._raw_data

    @staticmethod
    def read_raw_data(dfile):
        with open(dfile, 'r', encoding="ascii", errors="surrogateescape") as f:
            raw_data = f.readlines()
        return raw_data

    def copy(self):
        return deepcopy(self)