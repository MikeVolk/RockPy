from unittest import TestCase
import os
import RockPy
import numpy.testing as npt
import numpy as np
import pandas as pd

import logging

RockPy.log.setLevel('WARNING')


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
        f = self.m187a.f(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(f, s187a['f'], 2)

    def test_result_slope_vd(self):
        ms = (self.m187a,)
        res = ([[9.10E-01, -3.90E-01, -6.50E-01],
                [5.20E-01, -3.30E-01, -2.70E-01],
                [5.80E-01, -4.10E-01, -2.00E-01],
                [4.60E-01, -3.00E-01, -1.80E-01],
                [4.70E-01, -2.50E-01, -2.30E-01],
                [3.60E-01, -2.40E-01, -3.00E-02]],)
        for i, m in enumerate(ms):
            npt.assert_almost_equal(m.slope.vd(0, 700), res[i])

    def test_fvds_recipe_default(self):
        s187a = self.check.iloc[0]

        self.m187a.data.to_excel('/Users/mike/Desktop/pint.xls')
        res = self.m187a.fvds(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['fvds'], 2)

    def test_frac_recipe_default(self):
        s187a = self.check.iloc[0]

        self.m187a.data.to_excel('/Users/mike/Desktop/pint.xls')
        res = self.m187a.frac(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['FRAC'], 3)

    def test_q_recipe_default(self):
        s187a = self.check.iloc[0]

        res = self.m187a.q(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['q'], 1)

    def test_beta_recipe_default(self):
        s187a = self.check.iloc[0]

        res = self.m187a.beta(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['beta'], 1)

    def test_g_recipe_default(self):
        s187a = self.check.iloc[0]

        res = self.m187a.g(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['g'], 1)

    def test_gap_max_recipe_default(self):
        s187a = self.check.iloc[0]

        res = self.m187a.gapmax(vmin=s187a['Tmin'], vmax=s187a['Tmax'], blab=s187a['BLab'])
        self.assertAlmostEqual(res, s187a['GAP-MAX'], 1)
