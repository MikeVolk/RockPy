import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
import io
from copy import deepcopy

class Mpms(Ftype):
    def __init__(self, dfile, snames=None, dialect=None):
        super().__init__(dfile, snames=snames, dialect=dialect)