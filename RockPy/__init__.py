import logging.config
import os
import pkgutil

import RockPy

installation_directory = os.path.dirname(RockPy.__file__)
test_data_path = os.path.join(installation_directory, 'tests', 'test_data')

# unit handling
import pint
ureg = pint.UnitRegistry()
ureg.define('emu = 1e-3 ampere * meter ^2')
ureg.define('1 gauss = 1e-4 tesla')

import RockPy.core.file_io

from RockPy.core.study import Study
from RockPy.core.sample import Sample
from RockPy.core.measurement import Measurement
from RockPy.core.ftype import Ftype

import RockPy.ftypes
import RockPy.ftypes.tools

import RockPy.packages

from RockPy.core.utils import to_tuple, welcome_message

''' PLOTTING '''
<<<<<<< HEAD
from RockPy.tools.plotting import colorpalettes, ls, marker
=======
from RockPy.tools.plotting import colorpalettes, ls, marker, figsize
plotting_style = os.path.join(installation_directory, 'tools', 'RockPy.mplstyle')
>>>>>>> c48944f... minor edits/comments

from RockPy.tools.data import datamining

colors = colorpalettes['cat10']

''' LOGGING '''
from RockPy.core.utils import create_logger
create_logger(debug=False)

# create the RockPy main logger
log = logging.getLogger('RockPy')

# read the abbreviations.txt file
abbrev_to_classname, classname_to_abbrev = RockPy.core.file_io.read_abbreviations()

''' automatic import of all subpackages in packages and core '''

# create implemented measurements dictionary
implemented_measurements = RockPy.ftypes.tools.__implemented__(Measurement)
implemented_ftypes = RockPy.ftypes.tools.__implemented__(Ftype)

# todo create config file for RockPy
auto_calc_results = True

def debug_mode(on=False):
    create_logger(debug=on)
