# Import modules
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import NN

class PSO:
    
    def __init__(self, n_networks):
        """
        The PSO object contains an input n_networks which is the number of neural networks
        that are to be initialised.
        networks: is a list to store the initialised networks
        global_best_value: is initialised as infinity
        global_best_position: gets its shape from the Neural Network's getParams function
        global_best_yHat: is initialised at floating point 0. Useful for future plotting of graphs.
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
            informants = random.choices(self.networks, k=6) # 3 informants for each particle
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
        The move_particles method contains: the Intertia weight(a), 
        Cognitive(b), Social (c) and Informants (d) weights of the PSO algorithm which can be adjusted
        and affect directly the value of the velocity.
        There is an extra weight value (e) that is called the Jump and is used over the whole velocity.
        
        This method loops through a list of neural networks and stores the product of 
        of interia weight multiplied by network's velocity plus a random number multiplied 
        by the cognitive weight multiplied by the difference of the personal_best_position
        of the network and network's position plus the social weight into a random number
        multiplied by the difference of global_best_position of the networks and network's
        position plus the weighted value of the informants best position minus the network position
        in a variable called new_velocity. 
        
        This will be weighted by the jump value and then it ssigns the network's velocity 
        to this variable and calls the move function from the NeuralNetwork class. 
        """
        a = 0.5 # Intertia: proportion of velocity to be retained
        b = 0.8 # Cognitive/personal velocity: proportion of personal best to be retained
        c = 1 # Social velocity: proportion of the informants' best to be retained
        d = 0.9   # Global: proportion of global best to be retained
        e = 1   # Jump size of a particle
        
        for network in self.networks:
            new_velocity = (a*network.velocity) + (b*random.random())*\
            (network.personal_best_position - network.position) +\
            (c*random.random())*(network.informants_best_position - network.position) +\
            (d*random.random())*(self.global_best_position - network.position)
            network.velocity = e*new_velocity
            network.move()
        
    def optimise(self):
        """
        The optimise method contains all the methods (from the Swarm/PSO class and the NN/Particle) 
        needed to define an epoch/pass of the PSO algorithm (optimization of the ANN parameters).

        forward() and mse() define an epoch of a particle (a pass of inputs)
        set_personal_best(), set_informants(), set_informants_best(), set_global_best() and move_particles()
        define the optimization of the ANN parameters.

        To update the parameters of each particle (Neural Network: weights and biases)
        the optimise method loops through a list of neural networks and:
        w1: takes the first three numbers from network's position array which is then 
        reshaped to the dimensions of the NeuralNetwork object's W1 parameter
        w2: takes the next three numbers from network's position array which is then 
        reshaped to the dimensions of the NeuralNetwork object's W2 parameter
        b1: takes the 7th item from the array
        b2: takes the 8th item from the array
        """
        for network in self.networks:
            # Epoch for each Particle or NN
            network.forward()
            network.mse()

            # PSO algorithm methods
            self.set_personal_best()
            self.set_informants()
            self.set_informants_best()
            self.set_global_best()
            self.move_particles()

            # Update of weights
            W1 = network.position[0:6]
            W2 = network.position[6:12]
            network.W1 = np.reshape(W1,network.W1.shape) 
            network.W2 = np.reshape(W2,network.W2.shape)
            network.b1 = network.position[12:13]
            network.b2 = network.position[13]
            

if __name__ == "__main__":
    pso = PSO(10)
    n_iterations = 100
    error_list = [] # holds the mse values
    yHat = []
    
    
     # The start time to calculate how long the algorithm takes. 
    start = time.process_time()
    
    # Sets the number of starting iterations/epochs
    iterations = 0
    while(iterations < n_iterations):
    
        print(f"Iteration:{str(iterations)} Error:{pso.global_best_value}")
        pso.optimise() 
        error_list.append(pso.global_best_value) # adds the mse best value to create a plot later
        yHat = pso.global_best_yHat
        
        iterations +=1
        print(f"Iteration:{str(iterations)} Error:{pso.global_best_value}")
   
    #Prints the global_best_value and the time taken to execute the algorithm
    print(f"GlobalBest: {pso.global_best_position} iters: {iterations} GlobalBestVal: {pso.global_best_value}")
    print(f"------------------------ total time taken: {time.process_time() - start} seconds") 

    # Graphs for the global best output and best value 
    yHat = pso.global_best_yHat
    plt.figure()

    plt.plot(NN.y,"red",yHat,"blue")
    plt.xlabel("Input values")
    plt.ylabel("Output values")
    plt.title("Desired vs Predicted output")
    plt.show()

    fitness = error_list
    plt.figure()
    plt.plot(fitness)
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean square error")
    plt.title("Mean Square Error per iteration")
    plt.show()
