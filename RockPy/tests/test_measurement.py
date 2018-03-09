import RockPy
import os
from unittest import TestCase


class TestMeasurement(TestCase):
    def setUp(self):
        self.s = RockPy.Sample('test')
        self.m = self.s.add_measurement(mtype='hysteresis', ftype='vsm',
                                        fpath=os.path.join(RockPy.test_data_path, 'VSM', 'hys_vsm.001'),
                                        series=[('A', 1, 'a'), ('B', 2, 'b')])
        print('\n' + '-' * 100)

    def test_has_sval(self):
        self.assertTrue(self.m.has_sval(1))
        self.assertTrue(self.m.has_sval(2))
        self.assertTrue(self.m.has_sval([1, 2], 'all'))
        self.assertTrue(self.m.has_sval([1, 3], 'any'))
        self.assertTrue(self.m.has_sval([3], 'none'))

        self.assertFalse(self.m.has_sval(3))
        self.assertFalse(self.m.has_sval([1, 2, 3], 'all'))
        self.assertFalse(self.m.has_sval([4, 3], 'any'))
        self.assertFalse(self.m.has_sval([1, 2], 'none'))
