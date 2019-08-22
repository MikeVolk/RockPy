from unittest import TestCase
from RockPy.core.file_io import ImportHelper

import numpy as np

class TestImportHelper(TestCase):


    # first three cases need to return non, because reading the file requires at least sname, mtype, and ftype,
    # with a blank for the sgroup. Thus 4 elements separated by '_'are required in the filename.
    # only sample name
    test_file_minimal_fail = 't32.dat'
    # group and sample
    test_file_minimal_2_fail = 'sgroup_sname.dat'
    # group and sample and mtype
    test_file_minimal_3_fail = 'sgroup_sname_hys.dat'

    test_file_minimal_ok = '_sname_hys_vsm.dat'
    test_file_complete = 'sgroup_sname_hys_vsm.dat'

    # complex filename
    test_file_1 = '(sg1,sg2)_(s1,s2)_(hys,dcd)_vsm#19.94mg,2mm,3mm#(series1,32,h)_(series2,1,n)#preAF.dat'

    test_elements = ['snames', 'sgroups', 'mtypes', 'ftype', 'fpath', 'dialect', 'mass', 'massunit', 'height',
                     'heightunit', 'diameter', 'diameterunit', 'lengthunit', 'series', 'comment', 'additional',
                     'suffix']

    # def test_from_folder(self):
    #     self.fail()

    def test_from_file(self):

        for f in [self.test_file_minimal_fail, self.test_file_minimal_2_fail, self.test_file_1]:

            if f in [self.test_file_minimal_fail, self.test_file_minimal_2_fail, self.test_file_minimal_3_fail]:
                self.assertEqual(None, ImportHelper.from_file(f))
                continue
            test_ih = ImportHelper.from_file(f)

            # for a single file, there is only one element for each of the properties (test_elements)
            # Therefore, all test_elements have to have the shape (1,n), with n being >=0
            # (e.g. several series may have n > 1)
            # test for the shape in individual file import
            for elem in test_ih.__dict__.items():
                self.assertEqual(1, np.shape(elem[1])[0])


    # def test_extract_measurement_block(self):
    #     self.fail()
    #
    # def test_extract_sample_block(self):
    #     self.fail()

    def test_extract_series_block(self):
        series_block = self.test_file_1.split('#')[2]
        self.assertEqual([('series1',32,'h'),('series2', 1,'n')],ImportHelper.extract_series_block(series_block))
    #
    # def test_extract_add_dialect_block(self):
    #     self.fail()
    #
    # def test_from_dict(self):
    #     self.fail()
    #
    # def test_get_measurement_block(self):
    #     self.fail()
    #
    # def test_get_sample_block(self):
    #     self.fail()
    #
    # def test_get_series_block(self):
    #     self.fail()
    #
    # def test_get_add_block(self):
    #     self.fail()
    #
    # def test_new_filenames(self):
    #     self.fail()
    #
    # def test_getImportHelper(self):
    #     self.fail()
    #
    # def test_nsnames(self):
    #     self.fail()
    #
    # def test_nfiles(self):
    #     self.fail()
    #
    # def test__gen_dicts(self):
    #     self.fail()
    #
    # def test_gen_measurement_dict(self):
    #     self.fail()
    #
    # def test_gen_sample_dict(self):
    #     self.fail()
    #
    # def test_return_file_infos(self):
    #     self.fail()
