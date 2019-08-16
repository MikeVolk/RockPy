import logging.config
import os
import pkgutil

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)
test_data_path = os.path.join(installation_directory, 'tests', 'test_data')

# unit handling
import pint
ureg = pint.UnitRegistry()


import RockPy.core.file_io
from RockPy.core.study import Study
from RockPy.core.sample import Sample
from RockPy.core.measurement import Measurement

import RockPy.io
from RockPy.core.utils import to_tuple, welcome_message

''' PLOTTING '''
from RockPy.tools.plotting import colorpalettes, ls, marker

colors = colorpalettes['cat10']

''' LOGGING '''
from RockPy.core.utils import create_logger
create_logger(False)

# create the RockPy main logger
log = logging.getLogger('RockPy')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in Packages and core '''

# # import all RockPy.packages
__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages([installation_directory+'/Packages']):
    print(module_name)
    __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)

# create implemented measurements dictionary
implemented_measurements = {m.__name__.lower(): m for m in Measurement.inheritors()}

# todo create config file for RockPy
auto_calc_results = True

def debug_mode(on=True):
    create_logger(on)


if __name__ == '__main__':
    pass