import logging
import RockPy.core.measurement as measurement

''' LOGGING '''
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG,
                    datefmt='%I:%M:%S')

# create the RockPy main logger
log = logging.getLogger('RockPy')

log.debug('This is RockPy')
