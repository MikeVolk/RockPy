import RockPy
from RockPy.core.ftype import Ftype
import pandas as pd
import numpy as np
from RockPy.Packages.Mag.io.vsm import Vsm
from copy import deepcopy


class Agm(Vsm):
    standard_calibration_exponent = 0

    mtype_translation = {'Direct moment vs. field; Hysteresis loop\n': ('hys',),
                         'Direct moment vs. field; Initial magnetization; Hysteresis loop\n': ('hys',),
                         'Remanence curves:  DCD\n': ('dcd',),
                         'Remanence curves:  IRM + DCD\n': ('irm', 'dcd'),
                         'Direct moment vs. field; First-order reversal curves\n': ('forc',)}

    def __init__(self, dfile, snames=None, dialect=None, reload=False):

        # get the file infos first -> line numbers needed ect.
        mtype, header_end, segment_start, segment_widths, self.data_start, self.data_widths, \
        self.file_length = self.read_basic_file_info(dfile)

        self.mtype = self.mtype_translation[mtype]
        self.header = self.read_header(dfile, header_end)
        # print(self.header)

        Ftype.__init__(self, dfile, snames=snames, dialect=dialect, reload=reload)
        self.segment_header = self.read_segement_infos(dfile, mtype, header_end, segment_start, segment_widths)
