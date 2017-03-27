import RockPy
import os
from unittest import TestCase


class TestMeasurement(TestCase):
    def test_remove_from_clsdata(self):
        s = RockPy.Sample('test')
        m = s.add_measurement(mtype='hys', ftype='vsm',
                              fpath=os.path.join(RockPy.test_data_path, 'hys_vsm.001'))
        m.remove_from_clsdata(m.mid)
        self.assertEqual(len(m.clsdata), 0)
        self.assertEqual(len(m._clsdata), 0)
        self.assertEqual(len(m._mids), 0)
        self.assertEqual(len(m._sids), 0)
