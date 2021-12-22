from unittest import TestCase
from RockPy import Study, Sample

class TestStudy(TestCase):
    def setUp(self):
        self.S = Study()

    def create_samples(self):
        samples_to_add = ['S%i'%i for i in range(10)]
        slist = [self.S.add_sample(sname) for sname in samples_to_add]
        return samples_to_add, slist

    # def test_n_samples(self):
    #     self.fail()

    def test_samples(self):

        samples_to_add, slist = self.create_samples()

        for s in self.S.samples:
            self.assertIn(s, slist)

    def test_samplenames(self):
        samples_to_add, slist = self.create_samples()

        for s in self.S.samplenames:
            self.assertIn(s, samples_to_add)

        self.assertEqual(samples_to_add, list(self.S.samplenames))

    def test_sample_exists(self):
        samples_to_add, slist = self.create_samples()

        with self.assertRaises(TypeError) as cm:
            self.S.sample_exists()
        for sname, sobj in zip(samples_to_add, slist):
            self.assertEquals(sobj, self.S.sample_exists(sname=sname))
            self.assertEquals(sobj, self.S.sample_exists(sobj = sobj))

        self.assertFalse(self.S.sample_exists(sname= 'does_not_exist'))

        sobj_does_not_exist = Sample('non existent')
        self.assertFalse(self.S.sample_exists(sobj=sobj_does_not_exist))

    # def test_n_groups(self):
    #     self.fail()
    #
    # def test_groupnames(self):
    #     self.fail()
    #
    # def test_samplegroups(self):
    #     self.fail()
    #
    # def test_add_sample(self):
    #     self.fail()
    #
    # def test_add_samplegroup(self):
    #     self.fail()
    #
    # def test_remove_sample(self):
    #     self.fail()
    #
    # def test_remove_samplegroup(self):
    #     self.fail()
    #
    # def test_get_measurement(self):
    #     self.fail()
    #
    # def test_get_sample(self):
    #     self.fail()
    #
    # def test_get_samplegroup(self):
    #     self.fail()
    #
    # def test_import_folder(self):
    #     self.fail()
    #
    # def test_import_file(self):
    #     self.fail()
    #
    # def test_info(self):
    #     self.fail()
