from unittest import TestCase
import RockPy.core.utils as RPutils
from RockPy.core.utils import tuple2str, extract_tuple
from RockPy.core.utils import return_list_only_on_multiple


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
        self.assertEqual(('a',), extract_tuple('a'))
        self.assertEqual(('a', 'b'), extract_tuple('(a,b)'))
        self.assertEqual(('a', 'b'), extract_tuple('[a,b]'))


class TestTuple2str(TestCase):
    def test_tuple2str(self):

        self.assertEqual(tuple2str((2)), '2')
        self.assertEqual(tuple2str((1,2)), '(1,2)')
        self.assertEqual(tuple2str(('t', 2, 't')), '(t,2,t)')


class TestReturn_list_only_on_multiple(TestCase):
    def test_return_list_only_on_multiple(self):
        self.assertEqual(return_list_only_on_multiple([[1, 2]]), [1, 2])
        self.assertEqual(return_list_only_on_multiple([1, 2]), [1, 2])
        self.assertEqual(return_list_only_on_multiple([[1]]), 1)
        self.assertEqual(return_list_only_on_multiple([1]), 1)
        self.assertEqual(return_list_only_on_multiple(1), 1)