import numpy as np
from quask.core import Kernel
from . import BaseKernelOptimizer
from ..evaluator import KernelEvaluator
import itertools
from pyswarm import pso

class PsiOptimizer(BaseKernelOptimizer):
    
    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        super().__init__(initial_kernel, X, y, ke)


        
    def optimize(self):    
        
        def cost_function(solution):
            print("solution: "  + str(solution))
            solution = np.concatenate([solution, measurement_numpy, type_numpy])
            print("new_solution: "  + str(solution))
            candidate_solution=Kernel.from_numpy(solution,
                                           self.initial_kernel.ansatz.n_features,
                                           self.initial_kernel.ansatz.n_qubits,
                                           self.initial_kernel.ansatz.n_operations,
                                           self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                           shift_second_wire=True)
            fitness = self.ke.evaluate(candidate_solution, None, self.X, self.y)
            return fitness
            
            
        initial_population = list(self.initial_kernel.to_numpy())
        measurement_numpy = initial_population[-self.initial_kernel.ansatz.n_qubits - 1:-1]
        type_numpy = initial_population[-1:]
        initial_population = initial_population[:-self.initial_kernel.ansatz.n_qubits - 1]
        max_iter= 50
        ROTATIONS = list(a + b for a, b in itertools.product(["I", "X", "Y", "Z"], repeat=2))
        index_generator_range = list(range(0, len(ROTATIONS) - 1))
        qbit1_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 1))
        qbit2_range = list(range(0, self.initial_kernel.ansatz.n_qubits - 1))
        feature_range = list(range(0, self.initial_kernel.ansatz.n_features -1))
        bandwidth_range = list(np.arange(0,1.1,0.1))
        
        var_bound=  []
        i = 0
        
        for el in initial_population:
            if i == 0:
                var_bound.append(index_generator_range)
                i = i + 1
            elif i == 1:
                var_bound.append(qbit1_range)
                i = i + 1
            elif i == 2:
                var_bound.append(qbit2_range)
                i = i + 1
            elif i == 3:
                var_bound.append(feature_range)
                i = i + 1
            elif i == 4:
                var_bound.append(bandwidth_range)
                i = 0
                
        var_lbound = [0] * len(initial_population)
        var_ubound = [el[-1:] for el in var_bound]
        var_ubound = [el[0] for el in var_ubound]
        print("var_ubound: " + str(var_ubound))
        
        params_opt, f_opt = pso(func=cost_function,lb=var_lbound,ub=var_ubound,swarmsize=len(initial_population), maxiter=max_iter)
        
        best_solution = np.concatenate([params_opt, measurement_numpy, type_numpy])
        
        print("Parameters of the best solution : {params_opt}".format(params_opt=params_opt))
        print("Best_optimal_value:", f_opt)
        
        
        final_solution = Kernel.from_numpy(best_solution, 
                                           self.initial_kernel.ansatz.n_features,
                                           self.initial_kernel.ansatz.n_qubits,
                                           self.initial_kernel.ansatz.n_operations,
                                           self.initial_kernel.ansatz.allow_midcircuit_measurement,
                                           shift_second_wire=True)
 
        return final_solution 