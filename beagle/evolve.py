from beagle import Individual
import numpy as np
import random

def mutate( i, mutator ):
    # Should this be an Individual class method?
    return Individual( vector=mutator( i.vector ) )

def crossover( i1, i2 ):
    # Should this be an Individual class method?
    return np.array( [ random.choice( [ a, b ] ) for a, b in zip( i1.vector, i2.vector ) ] )

    
