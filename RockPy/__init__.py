import logging.config
import os
import pkgutil

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)
test_data_path = os.path.join(installation_directory, 'tests', 'test_data')

# unit handling
import pint
ureg = pint.UnitRegistry()
ureg.define('emu = 1 * (erg / gauss) = 1e-3 ampere /  metere ^2')

import RockPy.core.file_io

from RockPy.core.study import Study
from RockPy.core.sample import Sample
from RockPy.core.measurement import Measurement
from RockPy.core.ftype import Ftype

import RockPy.Ftypes
import RockPy.Ftypes.tools

import RockPy.Packages

from RockPy.core.utils import to_tuple, welcome_message

''' PLOTTING '''
from RockPy.Tools.plotting import colorpalettes, ls, marker

colors = colorpalettes['cat10']

''' LOGGING '''
from RockPy.core.utils import create_logger
create_logger(debug=False)

# create the RockPy main logger
log = logging.getLogger('RockPy')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in Packages and core '''

# create implemented measurements dictionary
implemented_measurements = RockPy.Ftypes.tools.__implemented__(Measurement)
implemented_ftypes = RockPy.Ftypes.tools.__implemented__(Ftype)

# todo create config file for RockPy
auto_calc_results = True

def debug_mode(on=True):
    create_logger(debug=on)
