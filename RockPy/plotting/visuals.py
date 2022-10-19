import matplotlib.pyplot as plt
import RockPy.plotting.features as features
from matplotlib.pyplot import get_cmap
from .utils import MidpointNormalize

def forc(d, x, y, ax=None):
    if ax is None:
        ax = plt.gca()

    features.forc_distributions(ax,d,x,y)

