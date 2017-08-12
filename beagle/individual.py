from collections import Counter
import random
import numpy as np

def matches( vector, a ):
    return [ i for i, e in enumerate( vector ) if e == a ]

def mutate( i, mutator ):
    return Individual( vector=mutator( i.vector ) )

def crossover( i1, i2 ):
    return np.array( [ random.choice( [ a, b ] ) for a, b in zip( i1.vector, i2.vector ) ] )

class Individual:
    """
    Class definition for Individual objects.

    An Individual describes a single potential solution within the problem vector space.

    Example::
    
        >>> vector = np.array( [ 1, 0, 1, 0 ] )
        >>> Individual( vector )
        Individual([1 0 1 0])

    """
    
    def __init__( self, vector ):
        """
        Create an `Individual` object.

        >>> vector = np.array( [ 1, 0, 1, 0 ] )
        >>> Individual( vector )
        Individual([1 0 1 0])

        Args:
            vector (ndarray(int)): A vector of integers, describing this particular potential solution.

        Returns:
            None
        """
        self.vector = vector
        self._score = None

    def fitness_score( self, fitness_function, use_saved_value=True ):
        """
        Returns the fitness score of this `Individual`, evaluated with a particular objective function.

        >>> vector = np.array( [ 1, 0, 1, 0 ] )
        >>> ind = Individual( vector )
        >>> objective_function = lambda x: sum( x )
        >>> ind.fitness_score( objective_function )
        2

        Args:
            fitness_function (function): The objective function, f(x), where x is the vector for this Individual.
            use_saved_value (optional:bool): The first time `fitness_score()` is called, the score is saved. 
                                             If `use_saved_value` is `True`, subsequent calls will return the saved value instead of recalculating f(x).
                                             To force recalculation of f(x), set `use_saved_value=False`.
                                             Default: `True`.
        Returns:
            (float): The fitness score of this `Individual`.
        """
        if not self._score or not use_saved_value:
            self._score = fitness_function( self.vector )
        return self._score

    @property
    def score( self ):
        """
        Returns the fitness score of this `Individual`, providing this has already been calculated by passing the objective function to `fitness_score( f(x) )`.
        If the score has not yet been evaluated, trying to access this attribute will raise an `AtttributeError`.

        >>> ind = Individual( np.array( [ 1, 0, 1, 0 ] ) )
        >>> objective_function = lambda x: sum( x )
        >>> ind.fitness_score( objective_function )
        2
        >>> ind.score
        2

        Args:
            None

        Returns:
            (float): The fitness score of this `Individual`.
        """
        if not self._score:
            raise AttributeError( 'The fitness score for this Individual has not yet been calculated' )
        else:
            return self._score

    def off_target( self, target ):
        """
        Returns the difference between the counts of appearances of integers for this `Individual` vector and a target count. For example, an `Individual` with vector `[1, 0, 1, 0]` contains `1` twice and `0` twice. If the target composition is `1` four times, and `0` none, this method will return the difference: `{1: -2, 0: 2}`.

        Example:
 
            >>> ind = Individual( np.array( [ 1, 0, 1, 0 ] ) )
            >>> target = { 1: 4, 0: 0 }
            >>> ind.off_target( target )
            {1: -2, 0: 2}

        """
        difference = {}
        count = dict( Counter( self.vector ).items() )
        for k, v in target.items():
            if k in count:
                difference[k] = count[k] - v
            else:
                difference[k] = - target[k]
        return difference

    def constrain( self, target ):
        difference = self.off_target( target )
        while not all( v == 0 for v in difference.values() ):
            too_many = [ k for k, v in difference.items() if v > 0 ]
            too_few  = [ k for k, v, in difference.items() if v < 0 ]
            i = random.choice( too_many )
            j = random.choice( too_few )
            self.vector[ random.choice( matches( self.vector, i ) ) ] = j
            difference = self.off_target( target )

    def __eq__( self, other ):
        return np.array_equal( self.vector, other.vector )

    def __lt__( self, other ):
        return self.score < other.score

    def __repr__( self ):
        to_return = "Individual({})".format(self.vector)
        return to_return
