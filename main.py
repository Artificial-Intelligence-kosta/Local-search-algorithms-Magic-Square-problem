# -*- coding: utf-8 -*-
"""
Main script
"""
import pandas as pd
import time
import os

from MagicSquare import MagicSquare
from Algorithms import *

def MonteCarloSimulation(algorithm,init_states,MC_iterations,plots_dir):
    """
    Performs Monte Carlo simulation to estimate the number of violated constraints
    for the given algoritm and initial state.
    
    Parameters
    -----
    algorithm: object from one of the classes in the 'Algorithms.py' script
    init_states: list of initial states (objects from 'MagicSquare' class)
    for the algorithm
    MC_iterations: number of iterations for Monte Carlo simulation
    
    Returns
    -----
    average number of violated constraints   
    """
    number_of_violated = []
    needed_iterations = []
    for i in range(MC_iterations):
        print("MC iteration: {}".format(i+1))
        # run 1 MC iteration
        state, num_of_violated, needed_iter = algorithm.run(init_states[i])
        # make a report
        if i == 0:
            algorithm.makeReport(os.path.join(plots_dir,algorithm.name))
        number_of_violated.append(num_of_violated)
        needed_iterations.append(needed_iter)
    # calculate scores
    avg_num_of_violated = sum(number_of_violated)/len(number_of_violated)
    std_num_of_violated = m.sqrt(sum([(x-avg_num_of_violated)**2 for x in number_of_violated])/len(number_of_violated))
    avg_iterations = sum(needed_iterations)/len(needed_iterations)
    std_iterations = m.sqrt(sum([(x-avg_iterations)**2 for x in needed_iterations])/len(needed_iterations))
    return avg_num_of_violated, std_num_of_violated, avg_iterations, std_iterations

def makeDirectoryTree(tree,output_dir):
    """
    Make directories from 'tree' in the 'otput_dir'.
    """
    for d in tree:
        os.makedirs(os.path.join(output_dir,d),exist_ok = True)
if __name__=="__main__":
    #np.random.seed(1) # comment this line if you want different initial state
    # Make directories for plots
    plots_dir = os.path.join(os.getcwd(),'plots')
    makeDirectoryTree(['plots'],os.getcwd())
    os.makedirs(plots_dir, exist_ok=True)
    # Initialize parameters
    n = 3
    iterations = 1000
    MC_iterations = 100
    # Create random initial states
    init_states = []
    for i in range(MC_iterations):
        init_state = MagicSquare(n)
        init_state.genRandSquare()
        init_states.append(init_state)
    # Create algorithms
    alg1 = RandomSearch(iterations)
    alg2 = HillClimbing(iterations)
    alg3 = SimulatedAnnealing(iterations, initial_temperature=10)
    alg4 = BeamSearch(3, iterations)
    alg5 = GeneticAlgorithm(10, 0.05, iterations)
    
    algorithms = [alg1,alg2,alg3,alg4,alg5]
    # create diectiories for plots
    makeDirectoryTree([alg.name for alg in algorithms], plots_dir)
    scores = pd.DataFrame(columns=["Algorithm", "Average Violated Constraints",
                                   "STD Violated Constraints", "Average Iterations",
                                   "STD Iterations","Execution Time",
                                   ])
    # Run the simulation
    for algorithm in algorithms:
         print('---'*10)
         print('Executing {} algorithm'.format(algorithm.name))
         tic = time.time()
         (avg_num_of_violated,std_num_of_violated,
          avg_iterations,std_iterations) = MonteCarloSimulation(algorithm,init_states,MC_iterations,plots_dir)
         toc = time.time()
         print("Average number of violated constraints: {:.3f}".format(avg_num_of_violated))
         print("Average number of needed iterations: {:.3f}".format(avg_iterations))
         print("Time: {:.3f}s".format(toc-tic))
         # save the data
         scores = scores.append(
             {"Algorithm": algorithm.name,
              "Average Violated Constraints": avg_num_of_violated,
              "STD Violated Constraints": std_num_of_violated,
              "Average Iterations": avg_iterations,
              "STD Iterations": std_iterations,
              "Execution Time": toc-tic}, 
             ignore_index = True
             )
    # save the scores
    scores.to_csv(os.path.join(os.getcwd(), "scores.csv"))
    
        
    