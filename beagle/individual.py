from collections import Counter
import random
import numpy as np

def matches( vector, a ):
    return [ i for i, e in enumerate( vector ) if e == a ]

class Individual:

    def __init__( self, vector ):
        self.vector = vector

    def fitness_score( self, fitness_function, *args ):
        return fitness_function( self.vector, *args )

    def off_target( self, target ):
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
        return self.vector == other.vector
