from unittest import TestCase
import RockPy

class TestResult(TestCase):

    def test_set_recipe(self):
        s = RockPy.Sample('paleointensity_test')
        m = s.add_simulation(mtype='pint')
