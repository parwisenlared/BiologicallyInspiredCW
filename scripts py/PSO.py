# Import modules
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import NN

class Particle():
    
    def __init__(self,SimplifiedNeuralNetwork):
        self.position = NN.SimplifiedNeuralNetwork.getParams
        self.velocity = np.zeros(NN.SimplifiedNeuralNetwork.getParams.shape)
        self.personal_best = np.zeros(NN.SimplifiedNeuralNetwork.getParams.shape)
        
    def move(self):
        self.position = self.position + self.velocity
        
def Solve(max_epochs,num_networks):
    networks = [NN.SimplifiedNeuralNetwork() for i in range(num_networks)]
    networks = NN.feedForward(networks)
    
    for network in networks:
        print(network)
#     print(networks)
        
g = np.array([0,0])
g

h = np.zeros(NN.nn1.getParams.shape)
h

Solve(10,10)

