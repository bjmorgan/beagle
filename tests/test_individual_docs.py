import unittest
import doctest
from beagle import individual

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(individual))
    return tests

if __name__ == '__main__':
    unittest.main()

