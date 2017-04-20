from unittest import TestCase
import os
import RockPy
import pandas as pd
import numpy.testing as npt

class TestPaleointensity(TestCase):
    def setUp(self):
        f187a = os.path.join(RockPy.test_data_path, 'TT format', '187A.tdt')
        self.s187a = RockPy.Sample(name='ET2_187A')
        self.m187a = self.s187a.add_measurement(mtype='paleointensity', ftype='tdt', fpath=f187a)

        self.check = pd.read_excel(os.path.join(RockPy.test_data_path, 'SPD_Test_Data_Table.xlsx'),
                                   index_col=0)
    def test_equal_acqu_demag_steps(self):
        th, ptrm = self.m187a.equal_acqu_demag_steps(vmin=150, vmax=300)
        self.assertLessEqual(150, th.index.max())
        self.assertLessEqual(150, ptrm.index.max())
        self.assertGreaterEqual(300, th.index.max())
        self.assertGreaterEqual(300, ptrm.index.max())
        npt.assert_almost_equal(th.index.values, ptrm.index.values)

    # def test_from_simulation(self):
    #     self.fail()
    #
    # def test_format_jr6(self):
    #     self.fail()
    #
    # def test_zf_steps(self):
    #     self.fail()
    #
    # def test_if_steps(self):
    #     self.fail()
    #
    # def test_ck(self):
    #     self.fail()
    #
    # def test_ac(self):
    #     self.fail()
    #
    # def test_tr(self):
    #     self.fail()
    #
    # def test_ifzf_diff_steps(self):
    #     self.fail()

    def test_slope_recipe_default(self):
        s187a = self.check.iloc[0]
        slope = self.m187a.slope(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(slope, s187a['b'], 3)

    def test_f_recipe_default(self):

        s187a = self.check.iloc[0]
        f = self.m187a.f(vmin=s187a['Tmin'],vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(f, s187a['f'], 2)


    def test_fvds_recipe_default(self):

        s187a = self.check.iloc[0]
        res = self.m187a.fvds(vmin=s187a['Tmin'],vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['fvds'], 2)