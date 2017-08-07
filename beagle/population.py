import random
from beagle import Individual
from scipy.constants import physical_constants
import numpy as np

k_b = physical_constants[ 'Boltzmann constant in eV/K' ][0]

class Population:

    def __init__( self, individuals=None ):
        if individuals:
            self.individuals = individuals
        else:
            self.individuals = []

    def __getitem__( self, i ):
        return self.individuals[i] 

    def __iter__( self ):
        return iter( self.individuals )

    def fitness_scores( self, fitness_function ):
        return np.array( [ i.fitness_score( fitness_function ) for i in self.individuals ] )

    @property
    def scores( self ):
        return np.array( [ i.score for i in self.individuals ] )

    def sample( self, n=1 ):
        return random.sample( self.individuals, n )

    def __add__( self, to_add ):
        if not isinstance( to_add, Individual ):
            raise TypeError
        self.individuals.append( to_add )
        return self 

    def __iadd__( self, to_add ):
        if not isinstance( to_add, Individual ):
            raise TypeError
        self.individuals.append( to_add )
        return self

    def __len__( self ):
        return len( self.individuals )

    def ranked( self, fitness_function ):
        return Population( individuals=sorted( self.individuals, key=lambda i: i.fitness_score( fitness_function ) ) )

    def boltzmann( self, fitness_function, temp, size ):
        s = self.fitness_scores( fitness_function )
        rel_p = np.exp( - ( s - s.min() ) / ( k_b * temp ) )
        rel_p /= sum( rel_p )
        return list( np.random.choice( self.individuals, size=size, p=rel_p, replace=False ) )

    def sort( self ):
        self.individuals.sort()
