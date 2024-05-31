import numpy as np
from quask.core import Kernel
from . import BaseKernelOptimizer
from ..evaluator import KernelEvaluator
import itertools
import random
from quask.optimizer import crossover_adhoc
import pygad

 
 
class GenOptimizer(BaseKernelOptimizer):
 
    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)
 
    
        
    def optimize(self):  
        
        #gene bounds
        ROTATIONS = list(a + b for a, b in itertools.product(["I", "X", "Y", "Z"], repeat=2))
        index_generator_range = list(range(0, len(ROTATIONS) - 1))
        qbit1_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 2))
        qbit2_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 2))
        feature_range = list(range(0, self.initial_kernel.ansatz.n_features -1))
        bandwidth_range = list(np.arange(0,1.1,0.1)) 
        
        def create_random_population(n_chromosomes): 
            init_pop = []
            temp = []
            for j in range(n_chromosomes):
                for i in range(self.initial_kernel.ansatz.n_operations):
                    rand_gen = random.randint(np.min(index_generator_range), np.max(index_generator_range))
                    rand_qbit_1 = random.randint(np.min(qbit1_range), np.max(qbit1_range))
                    rand_qbit_2 = random.randint(np.min(rand_qbit_1), np.max(qbit2_range))
                    rand_feature = random.randint(np.min(feature_range), np.max(feature_range))
                    rand_bandwidth = random.uniform(np.min(bandwidth_range), np.max(bandwidth_range))
                    individual = [rand_gen,rand_qbit_1,rand_qbit_2,rand_feature,rand_bandwidth]
                    temp.append(individual)
                temp=[sum(temp, [])]
                init_pop.append(temp[0])
                temp= []
            return init_pop
        
        
        def costruct_gene_space(genes_list):
            gene_space = []
            i = 0
            
            for el in genes_list:
                if i == 0:
                    gene_space.append(index_generator_range)
                    i = i + 1
                elif i == 1:
                    gene_space.append(qbit1_range)
                    i = i + 1
                elif i == 2:
                    gene_space.append(qbit2_range)
                    i = i + 1
                elif i == 3:
                    gene_space.append(feature_range)
                    i = i + 1
                elif i == 4:
                    gene_space.append(bandwidth_range)
                    i = 0  
                    
            return gene_space[:-self.initial_kernel.ansatz.n_qubits - 1]   
        
        def fitness_func(ga_instance, solution, solution_idx):
            solution = np.concatenate([solution, measurement_numpy, type_numpy])
            candidate_solution=Kernel.from_numpy(solution,
                                            self.initial_kernel.ansatz.n_features,
                                            self.initial_kernel.ansatz.n_qubits,
                                            self.initial_kernel.ansatz.n_operations,
                                            self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                            shift_second_wire=True)
            quantum_kernel_matrix,membership_matrix, fuzzyness, n_cluster = self.ke.evaluate(candidate_solution, None, self.X, self.y) # PER ORA TORNA I CLUSTER E LA MEMBERSHIP
            
            pow = np.power(membership_matrix, fuzzyness)
            j_KFCM = 0
            for i in range(n_cluster):
                row = []
                for k in range(len(self.X)):
                    j_KFCM += pow[k][i] * quantum_kernel_matrix[k][i] 
                
            print("j_KFCM: ",j_KFCM)
            
            if j_KFCM > 0 or j_KFCM< 0:
                return -1 * 1/(1+j_KFCM) ## QUA DEVO MINIMIZZARE J_KFMC
            else:
                return -1
    
        fitness_function = fitness_func
        num_generations = 50
        num_parents_mating = 2
        num_genes = len(self.initial_kernel.to_numpy()) - (self.initial_kernel.ansatz.n_qubits + 1)
        genes_list = list(self.initial_kernel.to_numpy())
    
 
    
        measurement_numpy = genes_list[-self.initial_kernel.ansatz.n_qubits - 1:-1]
        type_numpy = genes_list[-1:]
    
    
        gene_space = costruct_gene_space(genes_list)
        initial_population = create_random_population(50)
        parent_selection_type = "tournament"
        K_tournament = 3
        keep_parents = 1
        #crossover_type = "single_point"
        mutation_type = "random"
        #mutation_percent_genes = 50
        random_mutation_min_val = [0 for _ in range(num_genes)]
        random_mutation_max_val = [1 for _ in range(num_genes)] 
    
        ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        num_genes=num_genes,
                        initial_population=initial_population,
                        gene_type=float,
                        parent_selection_type=parent_selection_type,
                        K_tournament=K_tournament,
                        keep_parents=keep_parents,
                        crossover_type= "single_point",
                        crossover_probability=0.7,
                        mutation_type=mutation_type,
                        mutation_probability=0.2,
                        mutation_by_replacement=True,
                        #mutation_percent_genes=mutation_percent_genes,
                        random_mutation_min_val=random_mutation_min_val,
                        random_mutation_max_val=random_mutation_max_val,
                        gene_space=gene_space,
                        keep_elitism=1)
    
        ga_instance.run()
        ga_instance.plot_fitness(save_dir='./f_images_2/exp_')
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
        final_solution.to_qiskit_circuit()
        return final_solution 
    