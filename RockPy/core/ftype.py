import logging

import RockPy.core.utils


class Ftype(object):
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

        self.snames = RockPy.core.utils._to_tuple(snames)
        self.dfile = dfile
        self.dialect = dialect
        self.log().info('IMPORTING << %s , %s >> file: << %s >>' % (self.snames,
                                                                    type(self).__name__, dfile))
