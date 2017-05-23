from unittest import TestCase
from RockPy import Sample

class TestSample(TestCase):
    def setUp(self):
        self.s = Sample('TestSample')
        self.mlist = []
        for i in range(5):
            self.mlist.append(self.s.add_simulation(mtype='hysteresis', series=("No", i, '')))
            self.mlist.append(self.s.add_simulation(mtype='paleointensity', series=("#", i, '')))

    def test_get_measurement(self):
        print(self.mlist)
