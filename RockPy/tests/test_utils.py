from unittest import TestCase
from RockPy.core.utils import *


class Test_to_tuple(TestCase):
    def test__to_tuple(self):
        self.assertEqual((1,), to_tuple(1))
        self.assertEqual(('a',), to_tuple('a'))
        self.assertEqual(('a', 'b'), to_tuple(['a', 'b']))


class TestTuple2list_of_tuples(TestCase):
    def test_tuple2list_of_tuples(self):
        # test with list
        self.assertEqual([(1,)], tuple2list_of_tuples([1, ]))
        # test with item
        self.assertEqual([(1,)], tuple2list_of_tuples(1))
        # test with tuple
        self.assertEqual([(1,)], tuple2list_of_tuples((1,)))
        # test with list of tuples
        self.assertEqual([(1, 1), (2, 2)], tuple2list_of_tuples([(1, 1), (2, 2)]))


class TestExtract_tuple(TestCase):
    def test_extract_tuple(self):
        self.assertEqual(('a',), extract_tuple('a'))
        self.assertEqual(('a', 'b'), extract_tuple('(a,b)'))
        self.assertEqual(('a', 'b'), extract_tuple('[a,b]'))


class TestTuple2str(TestCase):
    def test_tuple2str(self):
        self.assertEqual(tuple2str((2)), '2')
        self.assertEqual(tuple2str((1, 2)), '(1,2)')
        self.assertEqual(tuple2str(('t', 2, 't')), '(t,2,t)')


class TestWelcome_message(TestCase):
    def test_welcome_message(self):
        welcome_message()


class TestConvert_units(TestCase):
    def test_convert_units(self):
        self.assertAlmostEqual(1, convert_units(1000, 'mg', 'g'))
        self.assertAlmostEqual(1, convert_units(1000, 'mT', 'T'))
        self.assertAlmostEqual(10, convert_units(1, 'mT', 'gauss'))
        self.assertAlmostEqual(2.20462, convert_units(1, 'kg', 'lbs'), delta=1e-3)
