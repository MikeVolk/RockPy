import logging
import logging.config
import os
import pkgutil, importlib

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)
test_data_path = os.path.join(installation_directory, 'tests', 'test_data')

import RockPy.core.measurement as measurement
import RockPy.core.file_io

from RockPy.core.study import Study
from RockPy.core.sample import Sample

from RockPy.core.utils import to_tuple
''' LOGGING '''

logging.config.fileConfig(os.path.join(installation_directory, 'logging.conf'))
# create the RockPy main logger
log = logging.getLogger('RockPy')

log.debug('This is RockPy')
log.debug('Installation dr: %s'%installation_directory)

import matplotlib
matplotlib.use('Qt5Agg')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in Packages and core '''

# # import all packages
__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages([installation_directory+'/Packages']):
    __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)

# create implemented measurements dictionary
implemented_measurements = {m.__name__.lower(): m for m in measurement.Measurement.inheritors()}

print('IMPLEMENTED MEASUREMENT TYPES     : \tFTYPES')
print('---------------------------------------------------------------------------')
print('\n'.join(['\t{:<26}: \t{}'.format(m, ', '.join(obj.ftype_formatters().keys()))
                 for m, obj in sorted(RockPy.implemented_measurements.items())]))
print()

log.info('creating Masterstudy. Can be used with ``RockPy.MasterStudy``')
RockPy.MasterStudy = RockPy.Study('MasterStudy')