from unittest import TestCase
import RockPy.core.file_io


class TestMinfo(TestCase):
    def setUp(self):
        self.only_sample = RockPy.core.file_io.ImportHelper(fpath='NoPath', sample='sample1', mtypes='hys', ftype='vsm')
        self.only_sample_mass = RockPy.core.file_io.ImportHelper(fpath='NoPath', sample='sample1', mtypes='hys', ftype='vsm',
                                                          mass='33mg')
        self.only_sample_mass_series = RockPy.core.file_io.ImportHelper(fpath='NoPath', sample='sample1', mtypes='hys',
                                                                 ftype='vsm',
                                                                 mass='33mg',
                                                                 series=('s1', 1, 'a'))
        self.only_sample_add = RockPy.core.file_io.ImportHelper(fpath='NoPath', sample='sample1', mtypes='hys', ftype='vsm',
                                                         add1=0)

    def test_extract_series(self):
        self.fail()

    def test_measurement_block(self):
        self.fail()

    def test_sample_block(self):
        self.fail()

    def test_series_block(self):
        self.fail()

    def test_add_block(self):
        self.fail()

    def test_comment_block(self):
        self.fail()

    def test_get_measurement_block(self):
        self.fail()

    def test_get_sample_block(self):
        self.fail()

    def test_get_series_block(self):
        self.fail()

    def test_get_add_block(self):
        self.fail()

    def test_is_readable(self):
        self.fail()

    def test_read_input(self):
        self.assertEqual((33, 'mg'), self.only_sample.read_input('33mg', 'kg'))
        self.assertEqual((33, 'mg'), self.only_sample.read_input((33, 'mg'), 'kg'))
        self.assertEqual((None, None), self.only_sample.read_input(None, 'kg'))
        self.assertEqual((33, 'kg'), self.only_sample.read_input(33, 'kg'))

    def test_fname(self):
        self.assertEqual('_sample1_HYS_VSM.000', self.only_sample.fname)
        self.assertEqual('_sample1_HYS_VSM#33.0mg.000', self.only_sample_mass.fname)
        self.assertEqual('_sample1_HYS_VSM#33.0mg#(s1,1,a).000', self.only_sample_mass_series.fname)
        self.assertEqual('_sample1_HYS_VSM###add1:0.000', self.only_sample_add.fname)

        fname = 'test_sample1_HYS_VSM###add1:0.000'
        self.assertEqual(fname, RockPy.core.file_io.minfo(fname).fname)

    def test_measurement_infos(self):
        self.fail()

    def test_sample_infos(self):
        self.fail()
