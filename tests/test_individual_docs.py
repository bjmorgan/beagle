import unittest
import doctest
from beagle import individual

testSuite = unittest.TestSuite() 
testSuite.addTest( doctest.DocTestSuite( individual ) )
unittest.TextTestRunner( verbosity=1 ).run( testSuite )

