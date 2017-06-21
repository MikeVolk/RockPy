from unittest import TestCase
import RockPy
from RockPy import Sample
import os

class TestSample(TestCase):
    def setUp(self):
        self.s = Sample('TestSample')
        self.mlist = []
        for i in range(2):
            self.mlist.append(self.s.add_measurement(
                fpath= os.path.join(RockPy.test_data_path, 'VSM', 'hys_vsm.001'),
                ftype='vsm',
                mtype='hysteresis', series=("No", i, '')))
            self.mlist.append(self.s.add_simulation(mtype='paleointensity', series=("#", i, '')))

    def test_get_measurement(self):

        # no arguments returns all measurements
        self.assertEqual(self.mlist, self.s.get_measurement())

        # specific mtype
        self.assertEqual([m for m in self.mlist if m.mtype == 'hysteresis'], self.s.get_measurement(mtype='hysteresis'))
        self.assertEqual([m for m in self.mlist if m.mtype == 'backfield'], self.s.get_measurement(mtype='backfield'))
        self.assertEqual([m for m in self.mlist if m.mtype == 'paleointensity'], self.s.get_measurement(mtype='paleointensity'))
        self.assertEqual(self.mlist, self.s.get_measurement(mtype=['paleointensity', 'hysteresis']))

        # invert
        self.assertEqual([], self.s.get_measurement(mtype=['paleointensity', 'hysteresis'], invert=True))

        # series
        self.assertEqual([m for m in self.mlist if 'No' in m.stypes], self.s.get_measurement(stype='No'))
        self.assertEqual([m for m in self.mlist if m.mtype == 'paleointensity'], self.s.get_measurement(stype='#'))
        self.assertEqual([m for m in self.mlist if 'test' in m.stypes], self.s.get_measurement(stype='test'))
        self.assertEqual([self.mlist[0]], self.s.get_measurement(series=('No', 0, '')))
        self.assertEqual([self.mlist[1]], self.s.get_measurement(series=('#', 0, '')))
        self.assertEqual([], self.s.get_measurement(series=('test', 0, '')))

    def test__del_mobj(self):
        mobj = self.mlist[1]

        # calculate the results
        self.s.calc_results()

        self.s._del_mobj(mobj)

        self.assertNotIn(mobj, self.s.measurements)
        self.assertNotIn(mobj.mid, self.s._results.index)
        self.assertNotIn(mobj.mid, self.s.results.index)
        self.assertNotIn(mobj.mid, mobj.__class__._clsdata[mobj.midx]['mid'])
        self.assertNotIn(mobj.mid, mobj.__class__._mids)