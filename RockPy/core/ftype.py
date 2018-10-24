import logging
import pandas as pd
import RockPy.core.utils


class Ftype(object):
    imported_files = {}

    def has_specimen(self, specimen):
        if specimen not in self.data['specimen'].values:
            Ftype.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Ftype.log().error('wrong sample name?')
            Ftype.log().error('These samples exist: %s' % set(self.data['specimens']))
            return False
        else:
            return True

    @classmethod
    def log(cls):
        # create and return a logger with the pattern RockPy.MTYPE
        return logging.getLogger('RockPy.%s' % cls.subclass_name().lower())

    def split_tab(self, line):
        return line.split('\t')

    @classmethod
    def subclass_name(cls):
        return cls.__name__

    def check_comments(self):
        """
        Checks for commented lines

        Parameters
        ----------
        dfile

        Returns
        -------

        """
        with open(self.dfile) as f:
            idx = [i for i,d in enumerate(f.readlines()) if d.startswith('#')]
        if idx:
            self.log().info('FILE contains {} comments. Lines {} were not read'.format(len(idx), idx))

    def __init__(self, dfile, snames=None, dialect=None, reload=False):
        """
        Constructor of the basic file type instance
        """

        # set file ID
        self.fid = id(self)

        # create sample name tuple
        self.snames = [str(i) for i in RockPy.core.utils.to_tuple(snames)] if snames else None
        self.dfile = dfile
        self.dialect = dialect

        if dfile not in self.__class__.imported_files or reload:
            self.log().info('IMPORTING << %s , %s >> file: << %s >>' % (self.snames,
                                                                        type(self).__name__, dfile))
            self.data = self.read_file()
            self.__class__.imported_files[dfile] = self.data.copy()
        else:
            self.log().info('LOADING previously imported file << %s , %s >> file: << %s >>\n\t>>> USE reload option if you want to read files from HD' % (self.snames,
                                                                        type(self).__name__, dfile))

            self.data = self.__class__.imported_files[dfile]

    def read_file(self):
        ''' Method for actual import of the file '''
        pass
