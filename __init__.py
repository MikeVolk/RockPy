import logging
import logging.config
import RockPy.core.measurement as measurement
import os

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)

''' LOGGING '''
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG,
#                     datefmt='%I:%M:%S')

logging.config.fileConfig(os.path.join(installation_directory, 'logging.conf'))
# create the RockPy main logger
log = logging.getLogger('RockPy')

log.debug('This is RockPy')

# create implemented measurements dictionary
implemented_measurements = {m.__name__.lower(): m for m in measurement.Measurement.inheritors()}
