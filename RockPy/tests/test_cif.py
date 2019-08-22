from unittest import TestCase
from RockPy.Ftypes.cif import Cif

class TestCif(TestCase):
    def test_write_header(self):
        tdata = {'sample_id': ' 1.0A',
                 'locality_id': 'erb',
                 'stratigraphic_level': 113.0,
                 'core_strike': 291.0,
                 'core_dip': 63.0,
                 'bedding_strike': 43.0,
                 'bedding_dip': 46.0,
                 'core_volume_or_mass': 1.0,
                 'comment': 'Sample just above tuff'}

        self.assertEqual(['erb  1.0A    Sample just above tuff\n','  113.0 291.0  63.0  43.0  46.0   1.0\n'],Cif.write_header(**tdata))