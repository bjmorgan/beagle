from collections import Counter
import random
import numpy as np

def matches(vector, a):
    """
    Returns indices where the elements of a vector match some value.

    Args:
        vector (ndarray(int)): A 1D numpy array describing a vector.
        a (int): The value to match.

    Returns:
        list(int): A list of indices for matching elements.

    Example::

        >>> vector = np.array( [ 1, 0, 1, 0 ] )
        >>> matches( vector, 0 )
        [1, 3]
    
    """
    return [ i for i, e in enumerate( vector ) if e == a ]

def mutate(i, mutator):
    """Return a new Individual, generated by mutating a starting Individual.

    Args:
        i (Individual): The Individual to be mutated.
        mutator (func): A function that takes a 1D numpy array as an argument,
            and returns a "mutated" new 1D numpy array.

    Returns:
        (Individual)

    Example::

        >>> vector = np.array( [ 1, 0, 1, 0 ] )
        >>> i = Individual( vector )
        >>> m = lambda x: 1-x # element-wise 1 <--> 0
        >>> mutate(i, m)
        Individual([0 1 0 1])

    """
    return Individual( vector=mutator( i.vector ) )

def crossover(i1, i2):
    return Individual(np.array([random.choice([a, b]) for a, b in zip(i1.vector, i2.vector)]))

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
        """Create an `Individual` object.

        Example::
            >>> vector = np.array( [ 1, 0, 1, 0 ] )
            >>> Individual( vector )
            Individual([1 0 1 0])

        Args:
            vector (ndarray(int)): A vector of integers, describing this particular potential 
                solution.

        Returns:
            None

        """
        self.vector = vector
        self._score = None

    def fitness_score( self, fitness_function, use_saved_value=True ):
        """Returns the fitness score of this `Individual`, evaluated with a particular objective function.

        Example::
            >>> vector = np.array( [ 1, 0, 1, 0 ] )
            >>> ind = Individual( vector )
            >>> objective_function = lambda x: sum( x )
            >>> ind.fitness_score( objective_function )
            2

        Args:
            fitness_function (function): The objective function, f(x), 
                where x is the vector for this Individual.
            use_saved_value (optional:bool): The first time `fitness_score()` is called, 
                the score is saved. If `use_saved_value` is `True`, subsequent calls will 
                return the saved value instead of recalculating f(x). To force recalculation 
                of f(x), set `use_saved_value=False`. Default: `True`.

        Returns:
            (float): The fitness score of this `Individual`.

        """
        if not self._score or not use_saved_value:
            self._score = fitness_function( self.vector )
        return self._score

    @property
    def score( self ):
        """Returns the fitness score of this `Individual`, providing this has already been 
        calculated by passing the objective function to `fitness_score( f(x) )`.
        If the score has not yet been evaluated, trying to access this attribute will 
        raise an `AtttributeError`.

        Example::
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
     
        Raises:
            AttributeError: If the score for this individual has not previously been evaluated.

        """
        if not self._score:
            raise AttributeError( 'The fitness score for this Individual has not yet been calculated' )
        else:
            return self._score

    def off_target( self, target ):
        """
        Returns the difference between the counts of appearances of integers for this 
        `Individual` vector and a target count. For example, an `Individual` with vector 
        `[1, 0, 1, 0]` contains `1` twice and `0` twice. If the target composition is `1` 
        four times, and `0` none, this method will return the difference: `{1: -2, 0: 2}`.

        Example:
 
            >>> ind = Individual( np.array( [ 1, 0, 1, 0 ] ) )
            >>> target = { 1: 4, 0: 0 }
            >>> output = ind.off_target( target )
            >>> output[0]
            2
            >>> output[1]
            -2

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
        """
        This method will attempt to constrain an `Individual` vector to match 
        the composition specified by `target`. Elements that appear with too high 
        frequency are replaced at random with elements that appear with too low frequency.

        Example:

            >>> ind = Individual( np.array( [ 1, 0, 2 ] ) )
            >>> target = { 0: 0, 1: 1, 2: 2 }
            >>> ind.constrain( target )
            >>> ind
            Individual([1 2 2])

        """
        difference = self.off_target( target )
        if len(self.vector) != sum(target.values()):
            raise ValueError("length mismatch between this individual and the target")
        while not all( v == 0 for v in difference.values() ):
            too_many = [ k for k, v in difference.items() if v > 0 ]
            too_few  = [ k for k, v in difference.items() if v < 0 ]
            i = random.choice( too_many )
            j = random.choice( too_few )
            self.vector[ random.choice( matches( self.vector, i ) ) ] = j
            difference = self.off_target( target )

    def __eq__( self, other ):
        """
        Test whether this `Individual` has the same vector as another `Individual`.

        Args:
            other (`Individual`): The other `Individual`.

        Returns:
            (bool): True | False.

        Example:

            >>> i = Individual( np.array( [ 1, 2, 3 ] ) )
            >>> j = Individual( np.array( [ 1, 2, 3 ] ) )
            >>> k = Individual( np.array( [ 2, 3, 1 ] ) )
            >>> i == j
            True
            >>> i == k
            False

        """
        return np.array_equal( self.vector, other.vector )

    def __lt__( self, other ):
        """
        Test whether this `Individual` has a score less than that of another `Individual`.

        Args:
            other (`Individual`): The other `Individual`.

        Returns:
            (bool): True | False.

        Example:

            >>> i = Individual( np.array( [ 1, 2, 3 ] ) )
            >>> j = Individual( np.array( [ 2, 2, 3 ] ) )
            >>> objective_function = lambda x: sum( x )
            >>> i.fitness_score( objective_function )
            6
            >>> j.fitness_score( objective_function )
            7
            >>> i < j
            True
            >>> j < i
            False

        """
        return self.score < other.score

    def __repr__( self ):
        to_return = "Individual({})".format(self.vector)
        return to_return
