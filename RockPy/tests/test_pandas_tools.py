from unittest import TestCase
import pandas as pd
from RockPy.tools.pandas_tools import *

class Test(TestCase):

    def test_dim2xyz(self):
        colD = 'D'
        colI = 'I'
        colM = 'M'
        colX = 'x'
        colY = 'y'
        colZ = 'z'

        df = pd.DataFrame(columns=['d','i','m'], data = [[0,90,1]])
        print(df[['d','i','m']])
        print(dim2xyz(df, colD='d',colI='i',colM='m'))
