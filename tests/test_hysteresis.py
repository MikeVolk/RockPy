from unittest import TestCase
import RockPy
import os

class TestHysteresis(TestCase):
    def setUp(self):
        self.s = RockPy.Sample('test')
        self.vsm01 = self.s.add_measurement(mtype='hys', ftype='vsm',
                              fpath=os.path.join(RockPy.test_data_path, 'hys_vsm.001'))
    def test_has_virgin(self):

        self.assertTrue(self.vsm01.has_virgin())
