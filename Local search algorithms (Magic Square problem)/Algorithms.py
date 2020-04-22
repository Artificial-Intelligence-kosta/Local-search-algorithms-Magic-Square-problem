# -*- coding: utf-8 -*-
"""
Script which contains classes of different search algorithms
"""
from MagicSquare import MagicSquare
import math as m
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

class RandomSearch(object):
    def __init__(self, iterations=1000):
        self.name = 'Random Search'
        self.iterations = iterations
        
    def run(self, init_state):
        """
        Run the Random Search algorithm for the 'self.iterations' number of
        iterations, with the initial state 'init_state'
        
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        
        Returns
        -----
        state: the state to which algorithm converged
        number of violated constraints for that state
        number of iterations needed to converge to the state
        """
        state = init_state
        i = 0
        while  i < self.iterations:
            if(state.getNumOfViolated() == 0): 
                break
            successor = state.getSuccessors(mode='random',k=1)[0]
            if (successor.getNumOfViolated() < state.getNumOfViolated()): # if successor is better than current
                state = successor # make successor current
            i += 1
        return state, state.getNumOfViolated(), i 
    def makeReport(self,output_dir):
        pass
    
# =============================================================================
class HillClimbing(object):
    def __init__(self, iterations=1000):
        self.name = 'Hill Climbing'
        self.iterations = iterations
        # used to make a report
        self.current_num_of_violated = []
    def run(self, init_state):
        """
        Run the Hill Climbing algorithm for the 'self.iterations' number of
        iterations, with the initial state 'init_state'
        
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        
        Returns
        -----
        state: the state to which algorithm converged
        number of violated constraints for that state
        number of iterations needed to converge to the state
        """
        state = init_state
        i = 0
        while  i < self.iterations:
            # make variables for report
            self.current_num_of_violated.append(state.getNumOfViolated())
            # check if global minimum is reached
            if(state.getNumOfViolated() == 0): 
                break
            # get successor
            successor = state.getSuccessors(mode='top',k=1)[0]
            if (successor.getNumOfViolated() >= state.getNumOfViolated()): # if we have reached minimum(local or global)
                self.current_num_of_violated.append(state.getNumOfViolated())
                break
            state = successor # make successor current
            i += 1
        return state, state.getNumOfViolated(), i
    def makeReport(self,output_dir):
        """
        Plot the number of violated constraints for solution through time.
        
        Parameters:
        -----
        output_dir: folder in which you want to save figures
        """
        fig = plt.figure(figsize=(15, 6))
        plt.xlabel('Time (iteration)')
        plt.ylabel('Number of violated constraints')
        plt.stem(self.current_num_of_violated,  use_line_collection = True)
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints.png'))
        plt.close(fig)
            
# =============================================================================
class SimulatedAnnealing(object):
    def __init__(self, iterations=1000, initial_temperature=100):
        """
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        initial_temperature: starting temperature for the algorithm
        """
        
        self.name = 'Simulated Annealing'
        self.T0 = initial_temperature
        self.iterations = iterations
        self.temperature = []
        self.probability = []
        self.current_num_of_violated = []
    def run(self, init_state):
        """
        Run the Simulated Annealing algorithm for the 'self.iterations' number of
        iterations, with the initial state 'init_state'
        
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        
        Returns
        -----
        state: the state to which algorithm converged
        number of violated constraints for that state
        number of iterations needed to converge to the state
        """
        state = init_state
        i = 0
        T = self.T0
        #delta_T = self.T0/self.iterations
        while  i < self.iterations:
            T = 0.995*T
            self.temperature.append(T) # save all temperatures to a list, for a later use
            self.current_num_of_violated.append(state.getNumOfViolated())
            if(state.getNumOfViolated() == 0) or (T == 0): 
                break
            successor = state.getSuccessors(mode='random',k=1)[0] # get random successor
            delta_E = state.getNumOfViolated() - successor.getNumOfViolated()
            if (delta_E > 0): # if successor is better
                state = successor
                self.probability.append(0)
            else:
                prob = m.exp(delta_E/T)
                self.probability.append(prob) # save all probabilities to a list, for a later use
                if (prob >= np.random.random()): # accept successor with some probability
                    state = successor
            i += 1
        return state, state.getNumOfViolated(), i
    def makeReport(self,output_dir):
        """
        Plot the number of violated constraints for solution, probability
        and temperature through time.
        
        Parameters:
        -----
        output_dir: folder in which you want to save figures
        """
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(3,1,1)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Temperature')
        plt.plot(self.temperature, figure=fig)
        plt.subplot(3,1,2)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Probability')
        plt.stem(self.probability, use_line_collection = True)
        plt.subplot(3,1,3)
        plt.xlabel('Time (iteration)')
        plt.ylabel('Violated constraints')
        plt.stem(self.current_num_of_violated,  use_line_collection = True)
        
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'AllPlots.png'))
        plt.close(fig)
        
# =============================================================================
class BeamSearch(object):
    def __init__(self, number_of_beams, iterations=1000):
        """
        Parameters
        -----
        number_of_beams: number of beams in every iteration of algorithm
        iterations: number of iterations you want algorithm to run for
        """
        self.name = 'Beam Search'
        self.iterations = iterations
        self.num_of_beams = number_of_beams
        self.avg_num_of_violated = []
        self.min_num_of_violated = []
    def run(self, init_state):
        """
        Run the Beam Search algorithm for the 'self.iterations' number of
        iterations, with the initial state 'init_state'.
        
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        
        Returns
        -----
        state: the state to which algorithm converged
        number of violated constraints for that state
        number of iterations needed to converge to the state
        """
        n = init_state.n
        states=[]
        states.append(init_state)
        i = 0
        while  i < self.iterations:
            # make varaibles for report
            minimum = 2*n+2+1
            avg = 0
            for state in states:
                avg += state.getNumOfViolated()/len(states)
                if (state.getNumOfViolated() < minimum):
                    minimum = state.getNumOfViolated()
            self.avg_num_of_violated.append(avg)
            self.min_num_of_violated.append(minimum)
            # check for solution
            converged = False
            for s in states:
                if (s.getNumOfViolated() == 0):
                    state = s
                    converged = True
                    break
            if converged:
                break
            # if no solution get successors
            successors = []
            for state in states: 
                successors += state.getSuccessors(mode='top',k=self.num_of_beams)
            # sort successors by the number of violated constraints
            successors = sorted(successors, key = lambda successor:successor.getNumOfViolated())
            # make unique states out of best successors
            states = []
            states.append(successors[0])
            j = 1
            while len(states) < self.num_of_beams:
                exist = False
                for state in states:
                    if np.array_equal(state.M,successors[j].M):
                        exist = True
                        break
                if not exist:
                    states.append(successors[j])
                j += 1       
            i += 1
            
        if (i == self.iterations):
            state = min(states, key = lambda state:state.getNumOfViolated())
        return state, state.getNumOfViolated(), i
    def makeReport(self,output_dir):
        """
        Plot the average and mimimal number of violated constraints for population,
        through time.
        
        Parameters:
        -----
        output_dir: folder in which you want to save figures
        """
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(2,1,1)
        plt.xlabel('Time (generation)')
        plt.ylabel('AVG violated constraints')
        plt.plot(self.avg_num_of_violated, figure=fig)
        plt.subplot(2,1,2)
        plt.xlabel('Time (generation)')
        plt.ylabel('MIN violated constraints')
        plt.plot(self.min_num_of_violated, figure=fig)
        
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints.png'))
        plt.close(fig)
        
# ============================================================================
class GeneticAlgorithm():
    def __init__(self, population_size,mutation_probability=0.05,iterations=1000):
        """
        Parameters
        -----
        population_size: size of the population for all generations
        mutation_probability: probability with which you perform mutation
        iterations: the number of iterations(generations) algorithm runs for
        """
        self.name = 'Genetic Algorithm'
        self.iterations = iterations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        # used to make a report
        self.avg_num_of_violated = []
        self.min_num_of_violated = []
    def run(self, init_state):
        """
        Run the Genetic Algorithm for the 'self.iterations' number of
        iterations, with the initial state 'init_state'.
        
        Parameters
        -----
        init_state: Magic Square object, which is initial state for the algorithm
        
        Returns
        -----
        state: the state to which algorithm converged
        number of violated constraints for that state
        number of iterations needed to converge to the state
        """
        n = init_state.n
        population = init_state.getSuccessors(mode='random',k=self.population_size)
        i = 0
        while  i < self.iterations:
            # make varaibles for report
            minimum = 2*n+2+1
            avg = 0
            for state in population:
                avg += state.getNumOfViolated()/len(population)
                if (state.getNumOfViolated() < minimum):
                    minimum = state.getNumOfViolated()
            self.avg_num_of_violated.append(avg)
            self.min_num_of_violated.append(minimum)
            # check if you have reached solution
            converged = False
            for s in population:
                if (s.getNumOfViolated() == 0):
                    state = s
                    converged = True
                    break
            if converged:
                break
            fitness = self._fitness(population)
            population = self._selection(population, fitness)
            children = self._crossover(population)
            population = self._mutation(children)
            i += 1
        if (i == self.iterations):
            state = min(population, key = lambda state:state.getNumOfViolated())
        return state, state.getNumOfViolated(), i
    
    def _fitness(self,population):
        """
        Calculate fitness function for each state in population.
        """
        n = population[0].n
        sum_of_all = sum(2*n+2-state.getNumOfViolated() for state in population)
        fitness = []
        if (sum_of_all == 0): # if all squares have mmaximum number of violated
            fitness = [1/len(population)]*len(population)
        else:
            for state in population:
                fitness.append((2*n+2-state.getNumOfViolated())/sum_of_all)
        return fitness
    def _selection(self,population,fitness):
        """
        Sample states from population with probabilities defined by their fitness.
        """
        population = np.random.choice(population,size=self.population_size, p=fitness, replace=True)
        return population
    def _crossover(self,parents):
        """
        Cross every 2 parents to get children
        Inversion of permutation method is implemented.
        The method is proposed in:
        'Genetic Algorithm Solution of the TSP Avoiding Special Crossover and Mutation'
        written by Göktürk Üçoluk
        """
        n = parents[0].n
        children = []
        i = 0
        while i  < (self.population_size/2)*2 - 1:
            inversion1 = self._getInversion(parents[i].M.reshape(n**2))
            inversion2 = self._getInversion(parents[i+1].M.reshape(n**2))
            ind = np.random.randint(low=0, high=n**2+1)
            child1_inverted = self._crossParents(inversion1,inversion2,ind)
            child2_inverted = self._crossParents(inversion2,inversion1,ind)
            child = MagicSquare(n)
            child.M = self._getPermutation(child1_inverted).reshape(n,n)
            children.append(child)
            child = MagicSquare(n)
            child.M = self._getPermutation(child2_inverted).reshape(n,n)
            children.append(child)
            i += 2
        return children
    def _crossParents(self,parent1,parent2,ind):
        """
        * Helper method for _crossover method.
        Make a child with 0..ind-1 elements from parent1, and ind..end elements
        from parent2
        
        """
        part1 = parent1[:ind]
        part2 = parent2[ind:]
        child = np.concatenate([part1,part2])
        return child
    def _getInversion(self,permutation):
        """
        * Helper method for _crossover method.
        Get inversion from the permutation sequence.
        If the inversion[i] = 0, there are no elements greater than i on the left of i.
        """
        inversion = []
        for i in range(len(permutation)):
            count = 0
            j = 0
            # go from left to right untill you reach element i, and count number of elements greater than i
            while (permutation[j] != i+1):
                if(permutation[j] > i+1):
                    count += 1
                j += 1
            inversion.append(count)
        return inversion               
    def _getPermutation(self,inversion):
        """
        * Helper method for _crossover method.
        Get permutation from the inversion sequence.
        If the inversion[i] = 0, there are no elements greater than i on the left of i.
        """
        permutation = [None]*len(inversion)
        positions = [None]*len(inversion)
        i = len(inversion)-1
        while i >= 0:
            positions[i] = int(inversion[i])
            for j in range(i+1,len(inversion)):
                if (positions[j] >= positions[i]):
                    positions[j] += 1
            i = i - 1
        for j,pos in enumerate(positions):
            permutation[pos] = j+1
        return np.array(permutation)
    def _mutation(self,children):
        """
        Performs swap mutation, mutation characteristic for permutation based encodings.
    
        Parameters
        -----
        children: list of children
        
        Returns
        -----
        mutated children
        """
        n = children[0].n
        for i in range(len(children)):
            prob = np.random.random()
            if (self.mutation_probability >= prob):
                x1,x2,y1,y2 = -1,-1,-1,-1
                while(x1 == x2) and (y1==y2):
                    x1,y1,x2,y2 = np.random.randint(0,n,size=4)
                children[i] = children[i].getSuccessor(x1,y1,x2,y2)
        return children
                    
    def makeReport(self,output_dir):
        """
        Plot the average and mimimal number of violated constraints for population,
        through time.
        
        Parameters:
        -----
        output_dir: folder in which you want to save figures
        """
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(2,1,1)
        plt.xlabel('Time (generation)')
        plt.ylabel('AVG violated constraints')
        plt.plot(self.avg_num_of_violated, figure=fig)
        plt.subplot(2,1,2)
        plt.xlabel('Time (generation)')
        plt.ylabel('MIN violated constraints')
        plt.plot(self.min_num_of_violated, figure=fig)
        
        if output_dir != None:
            plt.savefig(os.path.join(output_dir,'ViolatedConstraints.png'))
        plt.close(fig)    
    

