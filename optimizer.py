"""
Class that holds a genetic algorithm for evolving a network.
Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add 
import random
from network import Network

class Optimizer():
    # Class that implements genetic algorithm for MLP optimization

    def __init__(self,nn_param_choices,retrain=0.4,random_select=0.1,mutate_chance=0.2):

        """Create an optimizer.
        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated
        """
        self.mutate_chance = mutate_chance
        self.retrain = retrain
        self.random_select = random_select
        self.nn_param_choices = nn_param_choices

    def create_population(self,count):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
        pop=[]
        for i in range(0,count):
            network = Network(self.nn_param_choices)
            network.create_random()
            pop.append(network)
        return pop

    @staticmethod
    def fitness(network):
        return network.accuracy
    
    def grade(self,pop):
        """Find average fitness for a population.
        Args:
            pop (list): The population of networks
        Returns:
            (float): The average accuracy of the population
        """
        summed = reduce(add,(self.fitness(network) for network in pop))
        return summed/float(len(pop))

    def breed(self,mother,father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """
        children = []
        for i in range(2):
            child={}
            for param in self.nn_param_choices:
                child[param] = random_choice(
                    [mother.network[param],father.network[param]]
                )
            network = Network(self.nn_param_choices)
            network.create_set(child)

            if self.mutate_chance>random.random():
                network = self.mutate(network)
            children.append(network)
        return children
    
    def mutate(self, network):
        """Randomly mutate one part of the network.
        Args:
            network (dict): The network parameters to mutate
        Returns:
            (Network): A randomly mutated network object
        """
        #Choose a random key
        mutation = random.choice(list(self.nn_param_choices.keys()))
        # Muatate one of the params
        network.network[mutation] = random_choice(self.nn_param_choices[mutation])

        return network
    
    def evolve(self,pop):
        """Evolve a population of networks.
        Args:
            pop (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """
        graded = [(self.fitness(network),network) for network in pop]

        graded = [x[1] for x in sorted(graded, key=lambda x:x[0], reverse=True)]

        retrain_length = int(len(graded*self.retrain))

        parents = graded[:retrain_length]

        for individual in graded[retrain_length:]:
            if self.random_select>random.random():
                parents.append(individual)

        parents_length = len(parents)
        desired_length = len(pop)-parents_length
        children = []

        while len(children) < desired_length:
        
            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)
        return parents 
    
    