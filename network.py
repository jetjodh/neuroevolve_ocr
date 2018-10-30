import random
import logging
from train import train_and_score

class Network():
    """
    Represent a network and let us operate on it.
    Currently only works an MLP.
    """
    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0
        self.nn_param_choices = nn_param_choices
        self.network = {}

        def create_random(self):
            # Create random network
            for key in self.nn_param_choices:
                self.network[key] = random.choice(self.nn_param_choices[key])
        
        def create_set(self,network):
            self.network = network  
        
        def train(self, dataset):
            if self.accuracy == 0:
                self.accuacy = train_and_score(self.network, dataset)
        
        def print_network(self):
            logging.info(self.network)
            logging.info("Network accuracy: %.2f%%"%(self.accuracy*100))