import unittest
from unittest.mock import Mock
import numpy as np
from beagle import Individual

class TestIndividual( unittest.TestCase ):

    def setUp( self ):
        self.individual = Individual( np.array( [ 1, 2, 3 ] ) )

    def test_init( self ):
        v = np.array( [ 1, 2, 3 ] )
        i = Individual( vector=v )
        np.testing.assert_array_equal( i.vector, v )
        self.assertEqual( i._score, None )

    def test_fitness_score( self ):
        f = Mock( return_value=6 )
        score = self.individual.fitness_score( f )
        self.assertEqual( score, 6 )
        f.assert_called_with( self.individual.vector )

    def test_fitness_score_if_already_calculated( self ):
        self.individual._score = 3.5
        f = Mock()
        self.assertEqual( self.individual.fitness_score( f ), 3.5 )
    
    def test_off_target( self ):
        target = { 1:1, 2:2, 3:0 }
        self.assertEqual( self.individual.off_target( target ), { 1:0, 2:-1, 3:+1 } )
   
    def test_eq_returns_true_if_equal( self ):
        i = Individual( np.array( [ 1, 2, 3 ] ) )
        self.assertEqual( self.individual == i, True )

    def test_eq_returns_false_if_not_equal( self ):
        i = Individual( np.array( [ 3, 2, 1 ] ) )
        self.assertEqual( self.individual == i, False )
    
if __name__ == '__main__':
    unittest.main()
