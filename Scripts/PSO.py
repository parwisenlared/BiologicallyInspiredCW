# Import modules
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import NN
import NN_2

class PSO:
    
    def __init__(self, n_networks):
        """
        The PSO object contains an input n_networks which is the number of neural networks
        that are to be initialised.
        networks: is a list to store the initialised networks
        global_best_value: is initialised as infinity
        global_best_position: gets its shape from the Neural Network's getParams function
        yHat: is initialised at floating point 0. It is needed to plot a graph
        yHat_l: is a list to store the yHat values that is needed to plot a graph
        """
        self.n_networks = n_networks
        self.networks = [NN.NeuralNetwork(NN.x,NN.y) for i in range(self.n_networks)]
        self.global_best_value = float("inf")
        self.global_best_position = NN.NeuralNetwork(NN.x,NN.y).getParams.shape
        self.global_best_yHat = 0
    
    def set_personal_best(self):
        """
        The set_personal_best method loops through a list of networks, assisigns a 
        fitness_candidate which is the network's fitness. If the networks' 
        personal_best_value is greater that fitness_candidate; it then assigns the 
        personal_best_value as the fitness_candidate. It then updates the network's
        personal_best_position as the network's position.
        """
        for network in self.networks:
            if(network.personal_best_value > network.fitness):
                network.personal_best_value = network.fitness
                network.personal_best_position = network.position
                
    
    def get_personal_best(self):
        particles_position = []
        for network in self.networks:
            particles_position.append(network.position)
        return 
    
    # The variable informants is in each network, here I just create informants for each of them.
    def set_informants(self):
        for network in self.networks:
            informants = random.choices(self.networks, k=3) # 3 informants for each particle
            network.informants = informants
    
    # In this funcion I am instantiating the best_value of each informant in     
    def set_informants_best(self):
        for network in self.networks:
            for informant in network.informants:
                if(informant.personal_best_value > informant.fitness):
                    informant.informants_best_value = informant.fitness
                    informant.informants_best_position = informant.position
        
    def set_global_best(self):
        """
        The set_global_best method loops through a list of networks and assigns the 
        best_fitness_candidate to the network's fitness. If the global_best_value 
        is greater than the best_fitness_candidate the global_best_value is assigned as 
        best_fitness_candidate and the global_best_position becomes the network's position
        """
        for network in self.networks:
             if(self.global_best_value > network.personal_best_value):
                self.global_best_value = network.personal_best_value
                self.global_best_position = network.position
                self.global_best_yHat = network.yHat
                
        
    def get_global_best(self):
        print (f"Value:{self.global_best_value}, Position: {self.global_best_position}")
    
                
    def move_particles(self):
        """
        The move_particles method contains:
        
        the Intertia weight(W)
        Cognitive(c1)
        Social(c2) weights of the PSO algorithm which can be adjusted.
        
        This method loops through a list of neural networks and stores the product of 
        of interia weight multiplied by network's velocity plus a random number multiplied 
        by the cognitive weight multiplied by the difference of the personal_best_position
        of the network and network's position plus the social weight into a random number
        multiplied by the difference of global_best_position of the networks and network's
        position in a variable called new_velocity. It then assigns the network's velocity
        to this variable and calls the move function from NeuralNetwork class. 
        """
        a = 0.5 # Intertia
        b = 0.8 #Cognitive/personal velocity
        c = 0.9 # Social velocity
        d = 1 # informants
        e = 1 # Jump
        
        for network in self.networks:
            new_velocity = (a*network.velocity) + (b*random.random())*\
            (network.personal_best_position - network.position) +\
            (c*random.random())*(self.global_best_position - network.position) + \
            (d*random.random())*(network.informants_best_position - network.position)
            network.velocity = e*new_velocity
            network.move()
            
        # I added the Jump (the value is 1 by the pseudocode of the book they suggest, so does not affect)
        # but I think we do need to put it.
        
    def optimise(self):
        """
        The optimise method loops through a list of neural networks and:
        w1: takes the first three numbers from network's position array which is then 
        reshaped to the dimensions of the NeuralNetwork object's W1 parameter
        w2: takes the next three numbers from network's position array which is then 
        reshaped to the dimensions of the NeuralNetwork object's W2 parameter
        b1: takes the 7th item from the array
        b2: takes the 8th item from the array
        
        and uses these variables to forward propagate the neural network with these values.
        z2: is the dot product of input(x) and w1 plus bias(b1)
        a2: is the activation of the z2 using the activation function in NeuralNetwork class
        z3: is the dot product of a2 and W2 plus bias(b2)
        yHat: is the activation of the z3 using the activation function in NeuralNetwork class
        yHat_l: the yHat values are stored in a list for plotting graphs
        error: is calculated by using the Mean Square Error(mse) method using the target value(y)
        and predicted value(yHat). The network's fitness is updated using the error.
        """
        for network in self.networks:
         # by calling the methods here, the optimization is automatic and I do not need to call them outside.
         # just by calling PSO(num_NN) it is done.
    
            network.forward()
            network.mse()
            self.set_personal_best()
            self.set_informants()
            self.set_informants_best()
            self.set_global_best()
            self.move_particles()

            # Update of weights
            W1 = network.position[0:3]
            W2 = network.position[3:6]
            network.W1 = np.reshape(W1,network.W1.shape) 
            network.W2 = np.reshape(W2,network.W2.shape)
            network.b1 = network.position[6:7]
            network.b2 = network.position[7]
            

"""
pso1.optimise()
pso1.get_global_best()

plt.figure()
yHat1 = pso1.global_best_yHat
plt.plot(NN.y,"red",yHat1,"blue")
plt.title("y,yHat")
# plt.xlabel("Iterations")
# plt.ylabel("Errors")
plt.show()

"""
if __name__ == "__main__":
    pso = PSO(10)
    n_iterations = 100
    error_list = []
    yHat = 0
    # The start time to calculate how long the algorithm takes. 
    start = time.process_time()
    # Sets the number of starting iterations/epochs

    iterations = 0
    while(iterations < n_iterations):
    
        print(f"Iteration:{str(iterations)} Error:{pso.global_best_value}")
        pso.optimise()
        error_list.append(pso.global_best_value)
        iterations +=1
        print(f"Iteration:{str(iterations)} Error:{pso.global_best_value}")

    # Starting from the 1st iteration: prints the number of iterations, the global_best_value 
    # and the predicted(yHat) value. Also appends the global_best_value to the error_list
 

    #the global_best_value and the time taken to execute the algorithm
    print(f"GlobalBest: {pso.global_best_position} iters: {iterations} GlobalBestVal: {pso.global_best_value}")
    print(f"------------------------ total time taken: {time.process_time() - start} seconds") 


    # Show the graph
    yHat = pso.global_best_yHat
    plt.figure()

    plt.plot(NN.y,"red",yHat,"blue")
    plt.title("Desired vs Predicted output")
    plt.show()

    fitness = error_list
    plt.figure()
    plt.plot(fitness)
    plt.title("Mean Square Error")
    plt.show()