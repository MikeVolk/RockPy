import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from RockPy.plotting.utils import MidpointNormalize
from RockPy.plotting.core import feature

class forc_distributions(feature):

    def __init__(d, x, y, ax=None):
        super.__init__(ax=ax)
        self.d = d
        self.x =x
        self.y = y

    def __call__(self):
        self.ax.imshow(d*1e2,
              origin='lower',
              extent = (min(x), max(x), min(y), max(y)),
              aspect = 'equal',
              cmap=get_cmap('RdBu_r'),
              norm=MidpointNormalize(midpoint=0,vmin=d.min().min()*1e2, vmax=d.max().max()*1e2)
             )


def main():

    from magnetite_pressure import data

    forc_data = data.forc_data()
    print(forc_data)

if __name__ == '__main__':
    main()