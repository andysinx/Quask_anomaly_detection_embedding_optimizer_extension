from quask.core import Kernel
import numpy as np
from qiskit.circuit import  QuantumCircuit
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import pygad    
        
        
class GeneticQFKMeans():
    
    def __init__(self, x: np.float64, k: int, fuzzyness:int, n_qbit: int, quantum_kernel: Kernel):
        self.x = x
        self.fuzzyness = fuzzyness
        self.k = k
        self.quantum_kernel = quantum_kernel
        self.n_qbit=n_qbit
        
    @property
    def nqubits(self) -> int:
        return self.n_qbit

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the :class:`qiskit.circuit.QuantumCircuit` that implements the quantum feature map."""
        return self.feature_map

    @property
    def quantum_kernel(self) -> FidelityQuantumKernel:
        """Returns the :class:`qiskit_machine_learning.kernels.QuantumKernel` object of the QSVM model."""
        return self.quantum_kernel  
    
    
    #QKernel Based Fuzzy K-Means
    def kbfkmeans(self, max_iterations=100, epsilon=1e-4):
        
        #INIZIALIZZAZIONE
        def initialize_centroids():
            mu_j = np.random.rand(self.k, len(self.x[0]))
            return mu_j
        
        #ASSEGNAZIONE DEI PUNTI AL CLUSTER
        def initialize_membership_matrix():
            matrix = np.random.rand(self.k, len(self.x))
            matrix = np.round(matrix, decimals=8) 
            matrix /= matrix.sum(axis=1)[:, np.newaxis] 
            return matrix
        
        def calculate_quantum_distance_kernel_matrix(centroids):
            kernel_matrix = [[]]
            for i in range(len(self.x)):
                row = []
                for j in range(len(centroids)):
                    distance = self.quantum_kernel.kappa(self.x[i],centroids[j]) 
                    row.append(distance)
                kernel_matrix.append(row)

        def total(i, kernel_matrix):
            sum = 0
            for j in range(self.k):
                sum += 1 / (1 - kernel_matrix[i,j],self.fuzzyness - 1)
                
        
        def update_membership_matrix(membership_matrix,kernel_matrix):
            mem_matrix = membership_matrix.copy()
            for i in range(len(self.x)):
                for j in range(self.k):
                    mem_matrix[i,j] = pow(1 / (1 - kernel_matrix[i,j],self.fuzzyness - 1) / total(i, kernel_matrix))
                    
                    
            membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)  
            return membership_matrix
        
        
        #Aggiornamento centroidi cluster
        
        def weighted_point_to_cluster(mu,clust):
            tot_clust = list(clust)
            return mu/clust
        
        def update_centroids(membership_matrix,quantum_distance_calc):
            
            denominator = np.dot(np.power(membership_matrix.T, self.fuzzyness - 1), self.x)
            numerator = np.dot(denominator, quantum_distance_calc)
            denominator = np.sum(denominator, axis=0, keepdims=True)
            centroid = weighted_point_to_cluster(numerator/denominator)
            return centroid
        
        
        for _ in range(max_iterations):
            centroids = initialize_centroids()
            membership_matrix = initialize_membership_matrix()
            quantum_distance_calc= calculate_quantum_distance_kernel_matrix(centroids)
            membership_matrix = update_membership_matrix(membership_matrix, quantum_distance_calc, beta)
            new_centroids = update_centroids(membership_matrix, quantum_distance_calc)
            difference = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if difference < epsilon:
                return centroids, membership_matrix
            return centroids, membership_matrix
            
    # Genetic-QKernel Based Fuzzy K-Means        
    def gkbfkmeans(self, beta, max_iterations=50):
        
        #INIZIALIZZAZIONE
        def initialize_centroids():
            mu_j = np.random.rand(self.k, len(self.x[0]))
            return mu_j
        
        def initialize_membership_matrix():
            matrix = np.random.rand(self.k, len(self.x))
            matrix = np.round(matrix, decimals=8) 
            matrix /= matrix.sum(axis=1)[:, np.newaxis] 
            return matrix
        
        
        def calculate_quantum_distance_kernel_matrix(centroids):
            
            kernel_matrix = [[]]
            for i in range(len(self.x)):
                row = []
                for j in range(len(centroids)):
                    distance = self.quantum_kernel.kappa(self.x[i],centroids[j]) 
                    row.append(distance)
                kernel_matrix.append(row)
            return kernel_matrix

        def total(i, kernel_matrix):
            sum = 0
            for j in range(self.k):
                sum += 1 / (1 - kernel_matrix[i,j]) ** (self.fuzzyness - 1)
            return sum
                
        
        def update_membership_matrix(membership_matrix,kernel_matrix, beta):
            mem_matrix = membership_matrix.copy()
            for i in range(len(self.x)):
                for j in range(self.k):
                    mem_matrix[i,j] = pow(1 / (1 - kernel_matrix[i,j], self.fuzzyness - 1) / total(i, kernel_matrix))
            membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)  
            return membership_matrix
        
        
        #Aggiornamento centroidi cluster
        def weighted_point_to_cluster(num,den):
            return num/den
        
        def update_centroids(membership_matrix,quantum_distance_calc):
            
            denominator = np.dot(np.power(membership_matrix.T, self.fuzzyness - 1), self.x)
            numerator = np.dot(denominator, quantum_distance_calc)
            denominator = np.sum(denominator, axis=0, keepdims=True)
            centroid = weighted_point_to_cluster(numerator,denominator)
            return centroid
        
            
        
        centroids = initialize_centroids()
        membership_matrix = initialize_membership_matrix()
        quantum_distance_calc= calculate_quantum_distance_kernel_matrix(centroids)
        membership_matrix = update_membership_matrix(membership_matrix, quantum_distance_calc, beta)
        new_centroids = update_centroids(membership_matrix,quantum_distance_calc)
        
                
        gene_space = []
        for _ in range(self.k):
            # Creazione del sottolista con vettori di features impostate a 1
            list = [[1] * self.x[0]] * self.k
            gene_space.append(list)
            
        # Ottimizzazione col genetico
        def fitness_function(ga_instance, solution, solution_idx):
            def J_kfmc():
                matrix = []
                for k in range(self.k):
                    oper = []
                    for i in range(len(centroids)):
                        oper = 2 * (membership_matrix[k][i] ** self.fuzzyness) * (1 - quantum_distance_calc[k][i])
                    matrix.append(oper)
                return matrix
                    
            return -1 * (1/ 1 + J_kfmc)
            
            
          
        ga_instance = pygad.GA(num_generations=max_iterations,
                                num_parents_mating=2,
                                fitness_func=fitness_function,
                                num_genes=len(centroids[0]),
                                initial_population=centroids,
                                gene_type=float,
                                parent_selection_type='tournament',
                                K_tournament=3,
                                keep_parents=1,
                                crossover_type= 'uniform',
                                crossover_probability=0.7,
                                mutation_type='random',
                                mutation_probability=0.2,
                                mutation_by_replacement=True,
                                random_mutation_min_val=-1,
                                random_mutation_max_val=1,
                                gene_space=gene_space,
                                keep_elitism=1)
        
          
        ga_instance.run()
        ga_instance.plot_fitness(save_dir='./name_folder(DA_CAMBIARE!!!)/exp_')
    
        best_solution, best_solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=best_solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=best_solution_fitness))
            
        return best_solution, best_solution_fitness