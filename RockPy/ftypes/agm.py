from RockPy.core.ftype import Ftype
from RockPy.ftypes.vsm import Vsm


class Agm(Vsm):
    standard_calibration_exponent = 0

    mtype_translation = {'Direct moment vs. field; Hysteresis loop\n': ('hys',),
                         'Direct moment vs. field; Initial magnetization; Hysteresis loop\n': ('hys',),
                         'Remanence curves:  DCD\n': ('dcd',),
                         'Remanence curves:  IRM + DCD\n': ('irm', 'dcd'),
                         'Direct moment vs. field; First-order reversal curves\n': ('forc',)}

    # def __init__(self, dfile, snames=None, dialect=None, reload=False):
    #     super().__init__(dfile, snames=snames, dialect=dialect, reload=reload)
