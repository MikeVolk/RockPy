import logging
import logging.config
import os
import pkgutil

import matplotlib

matplotlib.use('Qt5agg')

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)
test_data_path = os.path.join(os.getcwd().split('RockPy')[0], 'RockPy', 'tests', 'test_data')

import RockPy.core.measurement as measurement
import RockPy.core.file_io

from RockPy.core.study import Study
from RockPy.core.sample import Sample


''' LOGGING '''

logging.config.fileConfig(os.path.join(installation_directory, 'logging.conf'))
# create the RockPy main logger
log = logging.getLogger('RockPy')

log.debug('This is RockPy')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in Packages and core '''
# get list of tupels with (package.name , bool)
subpackages = sorted([(i[1], i[2]) for i in pkgutil.walk_packages([installation_directory], prefix='RockPy.')])

# import all packages
for i in subpackages:
    # don't import testing packages
    if 'test' in i[0]:
        continue
    # store latest package name
    if i[1]:
        package = i[0]
        __import__(package)
    # the latest package name needs to be in the name of the 'non'-package to be imported
    if not i[1] and package in i[0]:
        # import the file in the package e.g. 'Packages.Mag.Visuals.paleointensity'
        __import__(i[0])


# create implemented measurements dictionary
implemented_measurements = {m.__name__.lower(): m for m in measurement.Measurement.inheritors()}

print('IMPLEMENTED MEASUREMENT TYPES     : \tFTYPES')
print('---------------------------------------------------------------------------')
print('\n'.join(['\t{:<26}: \t{}'.format(m, ', '.join(obj.ftype_formatters().keys()))
                 for m, obj in sorted(RockPy.implemented_measurements.items())]))
print()
