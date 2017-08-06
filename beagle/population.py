import random
from beagle import Individual

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
        return [ i.fitness_score( fitness_function ) for i in self.individuals ]

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

    def ranked( self, fitness_function ):
        return Population( individuals=sorted( self.individuals, key=lambda i: i.fitness_score( fitness_function ) ) )
