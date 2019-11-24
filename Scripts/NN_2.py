# Import modules
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time


# ----- ACTIVATION FUNCTIONS ------------------

def sigmoid(x):
    """Returns the sigmoid function for a given variable x"""
    return 1/(1+np.exp(-x))

def tanh(x):
    """Returns the tanH activation for a given variable x"""
    return np.tanh(x)

def cosine(x):
    """Returns the cosine activation for a given variable x"""
    return np.cos(x)

def gaussian(x):
    """Returns the gaussian activation for a given variable x"""
    return np.exp(-((x**2)/2))

def relu(X):
   return np.maximum(0,X)

# ----- INPUTS AND OUTPUTS ------------

#Sinus
data = np.loadtxt("Data/2in_xor.txt")
x = data[:, :2] # All input variables stored as x
y = data[:, 2:] # All test variables stored as y

# ----- NEURAL NETWORK ----------------

class NeuralNetwork(object):
    
    def __init__(self, x, y):
        """
        The NeuralNetwork object has a fitness attribute that is initialised at infinity
        and will will be updated
        
        The architecture of the network can be adjusted by changing the inputLayerSize,
        outputLayerSize, hiddenlayerSize. 
        
        The Network hyperparameters:
        -----------------------------
        W1: weights from the input layer to the hidden layer. Randomly initialised
        W2: weights from the hidden layer to the output layer. Randomly initialised
        a2_func: activation function of the hidden layer
        yHat_func: activation of the output layer
        b1: input to hidden bias
        b2: hidden to output bias
        
        The Particle Parameters:
        ------------------------
        position: uses the getParams function which yields an array of parameter values
        personal_best_position: is initialised as current position
        personal_best_value: is initialised at infinity
        self.velocity: is initialised at 0 in the shape of the position array
        """
        
        #Network architecture 
        self.inputLayerSize=2
        self.outputLayerSize=1
        self.hiddenlayerSize=3
        
        #Network hyperparameters
       self.W1 = np.random.randn(self.inputLayerSize, self.hiddenlayerSize)    # Weights for Input Layer
        self.W2 = np.random.randn(self.hiddenlayerSize, self.outputLayerSize)   # Weights for the outputs of the Hidden Layer
        self.a2_func = tanh     #Activation function for the Hidden Layer
        self.yHat_func = tanh   #Activation function for the output layer 
        self.b1 = random.random() #Bias 1
        self.b2 = random.random()   #Bias 2
        
        #Particle parameters
        self.position = self.getParams  # position of particle - 8 dimensions as it contains 6 weights and 2 biases
        self.personal_best_position = self.position # best postition of the particle so far
        self.personal_best_value = float("inf") # best fitness (min mse) of the particle so far
        self.velocity = np.zeros(self.getParams.shape)  # velocity of the particle with same dimensions as position
        self.informants = []    # array ocontaining the informants for the particle
        self.informants_best_value = float("inf")   # informant best value (min mse) - just one value evaluated over all informants best values
        self.informants_best_position = self.getParams  # informants best position - just one best value (caclulated over all informants positions)
        
        #Network input, outputs, fitness
        self.input = x
        self.output = y
        self.yHat = 0 # predicted output
        self.fitness = float("inf") # Infinite - easier to be compared in PSO algorithm
        
    
    def move(self):
        """The move function will change the particle position based on particle velocity"""
        self.position = self.position + self.velocity
     
    def __str__(self):
        """Returns a string representation of particle position and network fitness value"""
        return f"Position:{self.position}, Fitness:{self.fitness}"   
    
    def forward(self):
        """
        Forward propagation of the neural network 
        z2: is the dot product of input x and W1 plus bias(b1)
        a2: is the activation of the z2
        z3: is the dot product of a2 and W2 plus bias(b2)
        yHat: is the activation of the z3 - the predicted output
        """
        self.z2 = np.dot(self.input, self.W1) + self.b1       
        self.a2 = self.a2_func(self.z2)                 
        self.z3 = np.dot(self.a2, self.W2) + self.b2 
        self.yHat = self.yHat_func(self.z3)      
        return self.yHat 
    
    def mse(self):
        """ 
        Returns the value of the Mean Square Error of the predicted output compared with the true output of the function
        It is stored as the fitness of the Neural Network (fitness after feedforward) 
        """
        mse = np.square(np.subtract(self.output,self.yHat)).mean()
        self.fitness = mse
        return mse
    
    @property
    def getParams(self):
        """Returns the parameters of the neural network in an array that can be used in PSO"""
        array = self.W1.ravel()
        array = np.append(array,self.W2.ravel())
        array = np.append(array,self.b1)
        array = np.append(array,self.b2)
        return array


# Steps to make the neural network work: first call the method neural network, then forward, then fitness
# then call the variable network.fitness

#nn1 = NeuralNetwork(x,y)
#nn1.forward()
#nn1.mse() 

# call fitness: nn1.fitness


# Show the neural network predicted output compared to the real output

#plt.figure()
#y2 = nn1.yHat
#plt.plot(y,"red",y2,"blue")
#plt.title("y,yHat")
#plt.show()