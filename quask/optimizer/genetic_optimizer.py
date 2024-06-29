import numpy as np
from quask.core import Kernel
from . import BaseKernelOptimizer
from ..evaluator import KernelEvaluator
import itertools
import random
from quask.optimizer import crossover_adhoc
import pygad
import math

class GenOptimizer(BaseKernelOptimizer):
 
    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)
        
        
    def optimize(self):  
        
        
        #gene bounds
        ROTATIONS = list(a + b for a, b in itertools.product(["I", "X", "Y", "Z"], repeat=2))
        index_generator_range = list(range(0, len(ROTATIONS) - 1))
        print("index_generator_range", index_generator_range)
        qbit1_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 2))
        print("qbit1_range", qbit1_range)
        qbit2_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 2))
        print("qbit2_range", qbit2_range)
        feature_range = list(range(0, self.initial_kernel.ansatz.n_features -2))
        print("feature_range", feature_range)
        bandwidth_range = [1] #list(np.arange(0,1.1,0.1)) 
        
        def create_random_population(n_chromosomes): 
            init_pop = []
            temp = []
            for j in range(n_chromosomes):
                for i in range(self.initial_kernel.ansatz.n_operations):
                    rand_gen = random.randint(np.min(index_generator_range), np.max(index_generator_range))
                    print("rand_gen", rand_gen)
                    rand_qbit_1 = random.randint(np.min(qbit1_range), np.max(qbit1_range))
                    print("rand_qbit_1", rand_qbit_1)
                    rand_qbit_2 = random.randint(np.min(rand_qbit_1), np.max(qbit2_range))
                    print("rand_qbit_2", rand_qbit_2)
                    rand_feature = random.randint(np.min(feature_range), np.max(feature_range))
                    print("rand_feature", rand_feature)
                    rand_bandwidth = 1.0 #random.uniform(np.min(bandwidth_range), np.max(bandwidth_range))
                    individual = [rand_gen,rand_qbit_1,rand_qbit_2,rand_feature,rand_bandwidth]
                    temp.append(individual)
                    print("individual: ",individual)
                temp=[sum(temp, [])]
                print("temp: ",temp)
                init_pop.append(temp[0])
                temp= []
            return init_pop
        
        
        def costruct_gene_space(genes_list):
            gene_space = []
            i = 0
            
            for el in genes_list:
                if i == 0:
                    gene_space.append(range(max(index_generator_range)))
                    i = i + 1
                elif i == 1:
                    gene_space.append(range(max(qbit1_range)))
                    i = i + 1
                elif i == 2:
                    gene_space.append(range(max(qbit2_range)))
                    i = i + 1
                elif i == 3:
                    gene_space.append(range(max(feature_range)))
                    i = i + 1
                elif i == 4:
                    gene_space.append(range(max(bandwidth_range)))
                    i = 0  
                    
            return gene_space[:-self.initial_kernel.ansatz.n_qubits - 1]   
        
        '''def check_sol(solution):
            i = 0
            sol = solution.copy()
            sol = list(sol)
            print("sol_prima: ", sol)
           
            for i in range(len(sol)):
                if i == 0:
                    print("el: " ,  sol[i])
                    if sol[i] < np.min(index_generator_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(index_generator_range))
                        sol[i] = np.max(index_generator_range) %  sol[i]
                        print("qui1")
                    elif  sol[i] > np.max(index_generator_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(index_generator_range))
                        sol[i] =  sol[i] % np.max(index_generator_range) 
                        print("qui2")
                elif i == 1:
                    print("el: " ,  sol[i])
                    if  sol[i] < np.min(qbit1_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(qbit1_range))
                        sol[i] = np.max(qbit1_range) %  sol[i]
                        print("qui3")
                    elif  sol[i] > np.max(qbit1_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(qbit1_range))
                        sol[i] =  sol[i] % np.max(qbit1_range) 
                        print("qui4")
                elif i == 2:
                    print("el: " ,  sol[i])
                    if  sol[i] < np.min(qbit2_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(qbit2_range))
                        sol[i] = np.max(qbit2_range) %  sol[i] 
                        print("qui5")
                    elif  sol[i] > np.max(qbit2_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(qbit2_range))
                        sol[i] =  sol[i] % np.max(qbit2_range) 
                        print("qui6")
                elif i == 3:
                    print("el: " ,  sol[i])
                    if  sol[i] < np.min(feature_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(feature_range))
                        sol[i] = np.max(feature_range) %  sol[i]
                        print("qui7")
                    elif  sol[i] > np.max(feature_range):
                        print("el: " ,  sol[i])
                        print("min: " , np.min(feature_range))
                        sol[i] =  sol[i] % np.max(feature_range) 
                        print("qui8")
                elif i == 3:
                    i = 0 
            print("sol_dopo: ", sol)
            return np.array(sol)'''
            
        def fitness_func(ga_instance, solution, solution_idx):
            sol = solution #check_sol(solution)
            print("solution: " + str(sol))
            sol = np.concatenate([sol, measurement_numpy, type_numpy])
            print("SOL modif" + str(sol))
            candidate_solution=Kernel.from_numpy(sol,
                                            self.initial_kernel.ansatz.n_features,
                                            self.initial_kernel.ansatz.n_qubits,
                                            self.initial_kernel.ansatz.n_operations,
                                            self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                            shift_second_wire=True)
            print("SIAMO ALLA FITNESS")
            fitness = self.ke.evaluate(candidate_solution, None, self.X, self.y)
            print("fitness", fitness)
            return fitness
    
        fitness_function = fitness_func
        num_generations = 50
        num_parents_mating = 2
        num_genes = len(self.initial_kernel.to_numpy()) - (self.initial_kernel.ansatz.n_qubits + 1)
        genes_list = list(self.initial_kernel.to_numpy())
    
        print("genes_list: " + str(genes_list))
        print("genes_list len: " + str(len(genes_list)))
    
    
        measurement_numpy = genes_list[-self.initial_kernel.ansatz.n_qubits - 1:-1]
        type_numpy = genes_list[-1:]
        print("measurement numpy: " + str(measurement_numpy))
        print("type_numpy: " + str(type_numpy))
        
        gene_space = costruct_gene_space(genes_list)
        initial_population = create_random_population(35)
        print("initial_pop: ", initial_population)
        parent_selection_type = "tournament"
        K_tournament = 3
        keep_parents = 1
        mutation_type = "random"
        random_mutation_min_val = [0 for _ in range(num_genes)]
        print("Lung min val: ",len(random_mutation_min_val))
        random_mutation_max_val = [1 for _ in range(num_genes)] 
    
        
        print("num genes: " + str(num_genes))
        print("gene space: " + str(gene_space))
        print("gene space len: " + str(len(gene_space)))
        
        
        def sim_an_generate_neighbor(current_solution):
            
            current_sol = list(current_solution.copy())
            print("PRIMA FOR")
            curr_t_sol = []
            for i in range(0, len(current_sol), 5):
                curr_t_sol.append(current_sol[i:i + 5])
            curr_t_sol = list(map(list, curr_t_sol))
            print('curr_t_sol: ', curr_t_sol)
            
            
            new_solution = curr_t_sol
            print('new_solution: ', new_solution)
            print("INIZIO FOR")
            print("current_sol: ", current_sol)
            print("type_current_sol: ", type(current_sol)) 
            weights=[4, 2, 2, 4, 0] * int(len(new_solution[0])/5)
            print("weights: ", weights)
            num_chrom = random.choices(list(range(len(new_solution))), weights=[6] * len(new_solution), k=1)[0]
            allele_index = random.choices(list(range(len(new_solution[0]))), weights=weights, k=1)[0]
            print('num_chrom :', num_chrom )
            print('allele_index :', allele_index )
            new_value = 0
                
            #DA RIVEDERE
            if allele_index == 0:
                new_value = random.choice(index_generator_range)
            elif allele_index == 1:
                new_value = random.choice(qbit1_range)
            elif allele_index == 2:
                new_value = random.choice(qbit2_range)
            elif allele_index == 3:
                new_value = random.choice(feature_range)
            print("QUASI FINITO FOR")
            new_solution[int(num_chrom)][int(allele_index)] = new_value
            print("QUASI QUASI FINITO FOR")
            neighbor = [float(allele) for chrom in new_solution for allele in chrom]
            print("FINITO FOR")
            print("neigh: ", neighbor)
            
            return neighbor
        
        
        def flatten_list(lista):
            result = []
            for el in lista:
                if isinstance(el, list):
                    result.extend(flatten_list(el))  # Chiamata ricorsiva per espandere liste annidate
                else:
                    result.append(el)
            return result
        
        def memetic_siman_local_optimize(ga_instance) : 
            
            ga_instance.best_solutions = list(map(lambda x: x if not isinstance(x[0], list) else flatten_list(x), ga_instance.best_solutions))
            print("generations_completed: ", ga_instance.generations_completed)
                
            best_solution =  ga_instance.best_solutions[ga_instance.generations_completed-1]
            best_fitness = ga_instance.best_solutions_fitness[ga_instance.generations_completed-1]
            print("ga_instance.best_solutions: " , ga_instance.best_solutions)
            print("ga_instance.best_fitness_solutions: " , ga_instance.best_solutions_fitness)
            print("ga_instance.best_fitness_solutions_type: " , type(ga_instance.best_solutions_fitness))
                
            temp = 100.0
            max_iter = 50
            t = temp
            i = 0
                
            while i < len(range(max_iter)) and temp > 0 :
                print("FIN QUI OK")
                # Decrease temperature
                t /= float(i + 1)
                # Generate neighbour
                neighbor = sim_an_generate_neighbor(best_solution)
                print('neighbours: ', neighbor)
                print("DOPO GEN NEIGH")
                #Calculate new fitness value
                neighbors_fitness = [fitness_function(ga_instance=ga_instance, solution=neighbor, solution_idx=None)]
                print('neighbours_fitness: ', neighbors_fitness)

                # Accept the neighbor if it improves fitness or with a probability
                delta_fitness = [abs(neigh_fitn) - abs(best_fitness) for neigh_fitn in neighbors_fitness]
                print("delta_fitness: ", delta_fitness)
                delta_fitn_max = max(delta_fitness)
                indx_delta_fitness = delta_fitness.index(delta_fitn_max)
                    
                # Check if we should keep the new solution
                if delta_fitness[indx_delta_fitness] <= 0: 
                    best_solution = neighbor   
                    print("curr_sol_if: ", best_solution)
                    best_fitness = neighbors_fitness[indx_delta_fitness]
                    print("curr_fit_if: ", best_fitness)
                    # Add the best found solution to the new population
                    ga_instance.best_solutions.append(best_solution) 
                    ga_instance.best_solutions_fitness.append(best_fitness) 
                elif delta_fitness[indx_delta_fitness] > 0 and random.random() < math.exp(-delta_fitness[indx_delta_fitness] / t):
                    best_solution = neighbor 
                    print("curr_sol_elif: ", best_solution)
                    best_fitness = neighbor[indx_delta_fitness]
                    print("curr_fit_elif: ", best_fitness)
                    # Add the best found solution to the new population
                    ga_instance.best_solutions.append(best_solution) 
                    ga_instance.best_solutions_fitness.append(best_fitness) 
        
                print(f"Iteration {i}, Temperature {t:.3f}, Best Evaluation {best_fitness:.2f}")
                i = i + 1
                    
                print("PRIMA DELL APPEND")
                print("Fine APPEND")
                
               

        ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        num_genes=num_genes,
                        initial_population=initial_population,
                        gene_type=float,
                        parent_selection_type=parent_selection_type,
                        K_tournament=K_tournament,
                        keep_parents=keep_parents,
                        crossover_type= crossover_adhoc.heterogenous_crossover,
                        crossover_probability=0.7,
                        mutation_type=mutation_type,
                        mutation_probability=0.2,
                        mutation_by_replacement=True,
                        random_mutation_min_val=random_mutation_min_val,
                        random_mutation_max_val=random_mutation_max_val,
                        gene_space=gene_space,
                        on_generation=memetic_siman_local_optimize,
                        save_best_solutions=True,
                        keep_elitism=1)
    
        ga_instance.run()
        ga_instance.plot_fitness(savedir="./quask/plot_img/exp_.png")
    
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        solution = np.concatenate([solution, measurement_numpy, type_numpy])
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    
        final_solution = Kernel.from_numpy(solution, 
                                            self.initial_kernel.ansatz.n_features,
                                            self.initial_kernel.ansatz.n_qubits,
                                            self.initial_kernel.ansatz.n_operations,
                                            self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                            shift_second_wire=True)
    
        return final_solution
