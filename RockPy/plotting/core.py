import numpy as np
from RockPy.tools.plotting import get_ax

class figure():
    pass

class visual():
    pass

class feature():
    def __init__(self, ax=None):
        self.ax = get_ax(ax)

    def __call__(self):
        pass