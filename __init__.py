import logging
import logging.config
import RockPy.core.measurement as measurement
import os
import pkgutil

import RockPy
import RockPy.core.file_io

installation_directory = os.path.dirname(RockPy.__file__)

''' LOGGING '''
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG,
#                     datefmt='%I:%M:%S')

logging.config.fileConfig(os.path.join(installation_directory, 'logging.conf'))
# create the RockPy main logger
log = logging.getLogger('RockPy')

log.debug('This is RockPy')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in Packages and core '''
# get list of tupels with (package.name , bool)
subpackages = sorted([(i[1], i[2]) for i in pkgutil.walk_packages([os.path.dirname(RockPy.__file__)],
                                                                  prefix='RockPy.')])

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
