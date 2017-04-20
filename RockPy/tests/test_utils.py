from unittest import TestCase
import RockPy.core.utils as RPutils


class Test_to_tuple(TestCase):
    def test__to_tuple(self):
        self.assertEqual((1,), RPutils.to_tuple(1))
        self.assertEqual(('a',), RPutils.to_tuple('a'))
        self.assertEqual(('a', 'b'), RPutils.to_tuple(['a', 'b']))


class TestTuple2list_of_tuples(TestCase):
    def test_tuple2list_of_tuples(self):
        # test with list
        self.assertEqual([(1,)], RPutils.tuple2list_of_tuples([1, ]))
        # test with item
        self.assertEqual([(1,)], RPutils.tuple2list_of_tuples(1))
        # test with tuple
        self.assertEqual([(1,)], RPutils.tuple2list_of_tuples((1,)))
        # test with list of tuples
        self.assertEqual([(1, 1), (2, 2)], RPutils.tuple2list_of_tuples([(1, 1), (2, 2)]))


class TestExtract_tuple(TestCase):
    def test_extract_tuple(self):
        self.assertEqual(('a', 'b'), '(a,b)')
        self.assertEqual(('a', 'b'), '[a,b]')
