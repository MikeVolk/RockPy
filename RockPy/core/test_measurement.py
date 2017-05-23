from unittest import TestCase
from RockPy.core.measurement import Measurement
from RockPy.core.sample import Sample
import numpy as np


class TestMeasurement(TestCase):
    def setUp(self):
        self.s = Sample('test')
        self.m = Measurement(sobj=self.s)

    def test_add_series(self):
        series = ('test', 1, 'something')
        self.m.add_series(stype=series[0], sval=series[1], sunit=series[2])
        self.assertEqual(series, self.m.series[0])

        # if no unit is given None is added
        series = ('test', 1)
        self.m.add_series(stype=series[0], sval=series[1])
        self.assertEqual((series[0], series[1], None), self.m.series[1])

    def test_remove_Series(self):
        series = ('test', 1, 'something')
        self.m.add_series(stype=series[0], sval=series[1], sunit=series[2])
        self.assertEqual(series, self.m.series[0])
        self.m.remove_series(stype=series[0])
        self.assertEqual((None, np.nan, None), self.m.series[0])

    def test_get_series(self):
        m0 = Measurement(self.s, series=[('s1', 1, 'A'), ('s2', 1, 'b')])
        m1 = Measurement(self.s, series=[('s1', 3, 'A'), ('s2', 4, 'b')])
        self.assertEqual([('s1', 1, 'A')], m0.get_series(stype='s1'))
        self.assertEqual([('s1', 3, 'A')], m1.get_series(stype='s1'))
        self.assertEqual([('s1', 1, 'A'), ('s2', 1, 'b')], m0.get_series(sval=1))
        self.assertEqual([('s1', 3, 'A')], m1.get_series(sval=3))
        self.assertEqual([('s1', 3, 'A')], m1.get_series(series=[('s1', 3), ('s2', 1)]))

    def test_has_series(self):
        m0 = Measurement(self.s, series=[('s1', 1, 'A'), ('s2', 1, 'b')])
        self.assertTrue(m0.has_series([('s1', 1, 'A'), ('s2', 2, 'b')], method='any'))
        self.assertFalse(m0.has_series([('s1', 1, 'A'), ('s2', 2, 'b')], method='all'))
        self.assertTrue(m0.has_series([('s6', 3, 'C'), ('s4', 2, 'G')], method='none'))
        self.assertTrue(m0.has_series())

    def test_equal_series(self):
        m0 = Measurement(self.s, series=[('s1', 1, 'A'), ('s2', 1, 'b')])
        m1 = Measurement(self.s, series=[('s1', 1, 'A'), ('s2', 1, 'b')])
        m2 = Measurement(self.s, series=[('s1', 2, 'A'), ('s2', 1, 'b')])

        self.assertTrue(m0.equal_series(m1))
        self.assertFalse(m0.equal_series(m2))
        self.assertTrue(m0.equal_series(m2, ignore_stypes='s1'))