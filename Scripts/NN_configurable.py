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

def null(x):
    """Returns the null activation for a given variable x"""
    return x*0

def relu(X):
   return np.maximum(0,X)

# ----- INPUTS AND OUTPUTS ------------

#Sinus
data = np.loadtxt("Data/1in_linear.txt")
x = data[:, :1] # All input variables stored as x
y = data[:, 1:] # All test variables stored as y

# ----- NEURAL NETWORK ----------------

class NeuralNetwork(object):
    
    def __init__(self, x, y, neuron):
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
        b1: input to hidden layer bias
        b2: hidden to output layer bias
        
        The Particle Parameters:
        ------------------------
        position: uses the getParams function which yields an array of parameter values
        personal_best_position: is initialised as current position
        personal_best_value: is initialised at infinity
        velocity: is initialised at 0 in the shape of the position array
        informants: array ocontaining the informants of the particle
        informants_best_value: informant best value (min mse) - just one value evaluated over all informants best values
        informants_bes_position: informants best position - just one best value (caclulated over all informants positions)
        
        Other parameters of the network:
        --------------------------------
        input: holds the input data of the function given at initializing the NN
        output: holds the output data of the function given at initializing the NN
        yHat: holds the predicted output. Initialized to 0, once feedforward is called it will hold an array with the predicted values
        fitness: MSE between the predicted and the true output. Initialized to infinity (to ease the comparison).
        """
        
        #Network architecture
        self.inputLayerSize=1
        self.outputLayerSize=1
        #self.hiddenlayerSize= int(input("Inform the number of neurons in hidden layer of NN (particle): "))
        self.hiddenlayerSize = neuron
        
        #Network hyperparameters
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenlayerSize)    # Weights for Input Layer
        self.W2 = np.random.randn(self.hiddenlayerSize, self.outputLayerSize)   # Weights for the outputs of the Hidden Layer
        self.a2_func = relu     #Activation function for the Hidden Layer
        self.yHat_func = relu   #Activation function for the output layer 
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

        #Params
        self.pw1 = self.W1.ravel()
        self.pw2 = self.W2.ravel()
    
        
    
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
    
    def params(self):
        self.pw1 = self.W1.ravel()
        self.pw2 = self.W2.ravel()

    @property
    def getParams(self):
        """Returns the parameters of the neural network in an array that can be used in PSO"""
        array = self.W1.ravel()
        array = np.append(array,self.W2.ravel())
        array = np.append(array,self.b1)
        array = np.append(array,self.b2)
        return array


# Steps to make the neural network work: first call the method NeuralNetwork(x,y), then network.forward() and then network.mse()
# The fitness of the network will be accessed by network.fitness

#nn1 = NeuralNetwork(x,y, )
#nn1.forward()
#nn1.mse() 
#nn1.params()

#print(f"The following number is the fitness: {nn1.fitness}")
