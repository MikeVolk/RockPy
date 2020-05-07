#!/usr/bin/env python

# -*- coding: utf-8 -*-

import unittest
import test_utils


def suite():
    """Test suite"""

    suite = unittest.TestSuite()

    suite.addTests(
        [unittest.TestLoader().loadTestsFromTestCase(test_utils.Test_to_tuple),
         unittest.TestLoader().loadTestsFromTestCase(test_utils.TestExtract_tuple),
         unittest.TestLoader().loadTestsFromTestCase(test_utils.TestTuple2list_of_tuples),
         unittest.TestLoader().loadTestsFromTestCase(test_utils.TestTuple2str),
         unittest.TestLoader().loadTestsFromTestCase(test_utils.TestConvert_units),
         ]
    )

    return suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())
