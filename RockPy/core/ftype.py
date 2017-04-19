import logging
import pandas as pd
import RockPy.core.utils


class Ftype(object):
    imported_files = []
    _clsdata = pd.DataFrame(columns=('dfile',))

    def has_specimen(self, specimen):
        if specimen not in self.data['specimen'].values:
            Ftype.log().error('CANNOT IMPORT -- sobj_name not in ftype_data specimen list.')
            Ftype.log().error('wrong sample name?')
            Ftype.log().error('These samples exist: %s'%set(self.data['specimens']))
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

    def __init__(self, dfile, snames=None, dialect=None):
        """
        Constructor of the basic file type instance
        """
        self.fid = id(self)
        self.snames = RockPy.core.utils._to_tuple(snames) if snames else None
        self.dfile = dfile
        self.dialect = dialect

        if not dfile in self.__class__.imported_files:
            self.__class__.imported_files.append(dfile)
            self.log().info('IMPORTING << %s , %s >> file: << %s >>' % (self.snames,
                                                                        type(self).__name__, dfile))
            self.read_file()

    def read_file(self):
        ''' Method for actual import of the file '''
        pass