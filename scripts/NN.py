# Import modules
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
   # matplotlib inline


# ----- ACTIVATION FUNCTIONS ------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def cosine(x):
    return np.cos(x)

def gaussian(x):
    return np.exp(-((x**2)/2))

activations_list = [sigmoid,tanh,cosine,gaussian]
act_dict = {0.1:sigmoid, 0.2:tanh, 0.3:cosine, 0.4:gaussian}

f = np.random.choice(list(act_dict.keys()))
h = np.random.choice(list(act_dict.keys()))

h,f 


# ----- TESTING ------------

test_inputs = np.arange(-10,10,0.01)

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13,3))
axes[0].plot(test_inputs,sigmoid(test_inputs),linewidth=3)
axes[0].set_title("Sigmoid")
axes[1].plot(test_inputs,tanh(test_inputs),linewidth=3)
axes[1].set_title("Tanh")
axes[2].plot(test_inputs,cosine(test_inputs),linewidth=3)
axes[2].set_title("Cosine")
axes[3].plot(test_inputs,gaussian(test_inputs),linewidth=3)
axes[3].set_title("Gaussian")


# ------- INPUTS AND OUTPUTS ----------

df = pd.read_csv("../Data/1in_linear.txt", sep="\t", header=None)
df.columns = ["x","y"]
x = df["x"]
y = df["y"]

df.head()


# ------ INPUTS
data = np.loadtxt("../Data/1in_linear.txt")
data

# ------- SIMPLIFIED NEURAL NETWORK --------

    
class SimplifiedNeuralNetwork(object):
    
    def __init__(self):
        self.fitness = -1.
        
        self.inputLayerSize=1
        self.outputLayerSize=1
        self.hiddenlayerSize=3
                
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenlayerSize)
        self.W2 = np.random.randn(self.hiddenlayerSize, self.outputLayerSize)
        self.a2_func = np.random.choice(list(act_dict.keys()))
        self.yHat_func = np.random.choice(list(act_dict.keys()))
        
    def __str__(self):
        return f"Network:W1{self.W1}, Fitness:{self.fitness}"   
    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1) # Product of input layer and weights1
        self.a2 = act_dict[self.a2_func](self.z2) # Activation & z2  
        self.z3 = np.dot(self.a2, self.W2) # Product of a2 & weights2
        yHat = act_dict[self.yHat_func](self.z3) # Activation of z3
        return yHat        
    
    @property
    def getParams(self):
        array = self.W1.ravel()
        array = np.append(array,self.a2_func)
        array = np.append(array,self.W2.ravel())
        array = np.append(array,self.yHat_func)
        return array     
    

def rmse(predict, target):
    rmse_val = np.sqrt(np.subtract(target, predict)).mean()
    return rmse_val


def feedForward(networks):
    for network in networks:
        for col_val_x in df["x"]:
            yHat = network.forward(col_val_x)
            for col_val_y in df["y"]:
                error = rmse(col_val_y,yHat)
            network.fitness = error
    return networks

nn1 = SimplifiedNeuralNetwork()
print(nn1.getParams)

# ------- PSO -------------





