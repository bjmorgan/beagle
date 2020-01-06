import unittest
from unittest.mock import Mock, patch, call
import numpy as np
from beagle import Individual
from beagle.individual import matches, mutate, crossover

class TestIndividualFunctions( unittest.TestCase ):

    def test_matches_matches_equal_vectors( self ):
        vector = np.array( [ 1, 0, 1, 1, 0 ] )
        to_match = 1
        self.assertEqual( matches( vector, to_match ), [0, 2, 3] )

    def test_mutate_returns_mutated_individual(self):
        vector = np.array([ 1, 0, 1, 0 ])
        i = Individual(vector)
        m = lambda x: 1-x
        np.testing.assert_array_equal(np.array([0, 1, 0, 1]), mutate(i, m).vector)
  
    def test_crossover(self):
        v1 = np.array([1, 2, 3, 4])
        v2 = np.array([5, 6, 7, 8])
        i1 = Individual(v1)
        i2 = Individual(v2)
        with patch('beagle.individual.random.choice') as mock_random_choice:
            mock_random_choice.side_effect = [1, 6, 3, 8]
            i_co = crossover(i1, i2)
            self.assertEqual( i_co, Individual([1, 6, 3, 8]) )
            expected_calls = [call([1, 5]), call([2, 6]), call([3, 7]), call([4, 8])]
            mock_random_choice.assert_has_calls(expected_calls)

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

    def test_fitness_score_is_recalculated_if_use_saved_value_is_false( self ):
        self.individual._score = 3.5
        f = Mock( return_value = 4.0 )
        score = self.individual.fitness_score( f, use_saved_value=False )
        self.assertEqual( score, 4.0 )
        f.assert_called_with( self.individual.vector )
    
    def test_off_target( self ):
        target = { 1:1, 2:2, 3:0 }
        self.assertEqual( self.individual.off_target( target ), { 1:0, 2:-1, 3:+1 } )
   
    def test_eq_returns_true_if_equal( self ):
        i = Individual( np.array( [ 1, 2, 3 ] ) )
        self.assertEqual( self.individual == i, True )

    def test_eq_returns_false_if_not_equal( self ):
        i = Individual( np.array( [ 3, 2, 1 ] ) )
        self.assertEqual( self.individual == i, False )
    
    def test_score( self ):
        self.individual._score = 2.3
        self.assertEqual( self.individual.score, 2.3 )

    def test_score_raises_attribute_error_if_not_set( self ):
        with self.assertRaises( AttributeError ):
            self.individual.score

    def test_constrain( self ):
        self.individual.vector = np.array( [ 1, 0, 2 ] )
        self.individual.off_target = Mock( side_effect=[ { 0: 1, 1: 0, 2: -1 }, { 0: 0, 1: 0, 2: 0 } ] )
        self.individual.constrain( target={ 0:0, 1:1, 2:2 } )
        np.testing.assert_array_equal( self.individual.vector, np.array( [ 1, 2, 2 ] ) )

    def test_eq_returns_true_if_vectors_are_equal( self ):
        other_individual = Individual( self.individual.vector )
        self.assertEqual( self.individual == other_individual, True )


    def test_eq_returns_false_if_vectors_are_not_equal( self ):
        other_individual = Individual( np.array( [ 3, 2, 1 ] ) )
        self.assertEqual( self.individual == other_individual, False )

    def test_lt_returns_true_if_score_is_less_than( self ):
        self.individual._score = 5
        other_individual = Individual( np.array( [ 3, 2, 1 ] ) )
        other_individual._score = 6
        self.assertEqual( self.individual < other_individual, True )

    def test_lt_returns_false_if_score_is_not_less_than( self ):
        self.individual._score = 5
        other_individual = Individual( np.array( [ 3, 2, 1 ] ) )
        other_individual._score = 5
        self.assertEqual( self.individual < other_individual, False )

    def test_repr( self ):
        self.assertEqual( str( self.individual ), 'Individual([1 2 3])' )

if __name__ == '__main__':
    unittest.main()
