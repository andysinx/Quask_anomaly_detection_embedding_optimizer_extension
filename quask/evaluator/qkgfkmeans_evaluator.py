from quask.evaluator import KernelEvaluator, GeneticQFKMeans
from quask.core import Kernel
import numpy as np

class GeneticQFKMeansEvaluator(KernelEvaluator):
    
    def __init__(self): pass

    def evaluate(self, kernel: Kernel, K:np.ndarray, X: np.ndarray, y: np.ndarray):
        """Evaluate the current kernel and return the corresponding wrong labels.
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :return: wrong_labels """

        n = len(X)
        X_train = X[:n//2]
        X_val = X[n//2:]
        y_train = y[:n//2]#y[0]
        y_val = y[n//2:]#y[1:n]
        print("train_set",X_train[0].shape)
        model = GeneticQFKMeans(x=X, k=3, n_qbit=4, quantum_kernel=K)
        centroids, membership_matrix = model.kbfkmeans(4)
        print("centroids_final: ",centroids)
        print("membership_matrix_final: ", membership_matrix)
        return "ciao"
  
    def evaluate_test(self, kernel: Kernel, K:np.ndarray, X: np.ndarray, y: np.ndarray):
        pass
