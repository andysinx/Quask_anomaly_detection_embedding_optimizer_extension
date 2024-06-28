from sklearn.svm import OneClassSVM
import warnings
import numpy as np
from quask.core import Kernel
from quask.evaluator import KernelEvaluator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit import QuantumCircuit
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")



class OneClassQSVM(OneClassSVM):
  def __init__(self, kernel: Kernel):
    super().__init__(kernel="precomputed", nu=0.01, tol=1e-9)
    self._nqubits = 10
    self._featuremap = kernel.to_qiskit_circuit()
    self._quantum_kernel = FidelityQuantumKernel(feature_map=self._featuremap)
    self._kernel_matrix_train = None
    self._kernel_matrix_test = None
    self._train_data = None

  @property
  def kernel_matrix_train(self):
    return self._kernel_matrix_train

  @property
  def nqubits(self) -> int:
    return self._nqubits

  @property
  def feature_map(self) -> QuantumCircuit:
    """Returns the :class:`qiskit.circuit.QuantumCircuit` that implements the quantum feature map."""
    return self._feature_map

  @property
  def feature_map_name(self) -> str:
    """Returns the quantum feature map name."""
    return self._feature_map_name

  @property
  def quantum_kernel(self) -> FidelityQuantumKernel:
    """Returns the :class:`qiskit_machine_learning.kernels.QuantumKernel` object of the QSVM model."""
    return self._quantum_kernel

  def fit(self, train_data: np.ndarray, train_labels=None):
    self._train_data = train_data
    self._kernel_matrix_train = self._quantum_kernel.evaluate(train_data)
    self._kernel_matrix_train = self._kernel_matrix_train.astype(float)
    super().fit(self._kernel_matrix_train)

  def predict(self, x: np.ndarray, input_is_matrix: bool = False) -> np.ndarray:
    if input_is_matrix:
        test_kernel_matrix = x
    else:
        test_kernel_matrix = self._quantum_kernel.evaluate(x_vec=x, y_vec=self._train_data)
        test_kernel_matrix = test_kernel_matrix.astype(float)
        y = super().predict(test_kernel_matrix)
        y[y == 1] = 0
        y[y == -1] = 1
        return y

  def score(self, x: np.ndarray, y: np.ndarray, train_data: bool = False, sample_weight: np.ndarray = None,) -> float:
     if train_data:
        y_pred = self.predict(x)
        y = np.ones(len(x))  # To compute the fraction of outliers in training.
        return accuracy_score(y, y_pred, sample_weight=sample_weight)
     y_pred = self.predict(x)
     return accuracy_score(y.tolist(), y_pred.tolist(), sample_weight=sample_weight)

  def decision_function(self, x_test: np.ndarray) -> np.ndarray:
     test_kernel_matrix = self._quantum_kernel.evaluate(x_vec=x_test,y_vec=self._train_data)
     self._kernel_matrix_test = test_kernel_matrix
     return -1.0 * super().decision_function(test_kernel_matrix)
 
 
class OneClassQSVMEvaluator(KernelEvaluator):
  
    def __init__(self, args: dict):
                self.args = args

    def evaluate(self, kernel: Kernel, K:np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding wrong labels.
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :return: wrong_labels
        """
        
        print("len X: ", len(X))
        print("len y: ", len(y))
        print("y: " , y)
        y_1 = list(filter(lambda x: x == 1, y)) #1
        print("len y_1: ", len(y_1))
        y_2 = list(filter(lambda x: x == 0, y)) #0
        print("len y_2: ", len(y_2))
        print("y_1: ", y_1)
        print("y_2: ", y_2)
        
        if int(len(y_1)/2) % 2 == 0: 
          fst_part_y1 = y_1[:int(len(y_1)/2)]
          fst_part_y2 = y_2[:int(len(y_2)/2)]
          snd_part_y1 = y_1[int(len(y_1)/2):]
          snd_part_y2 = y_2[int(len(y_2)/2):]
        else:
          fst_part_y1 = y_1[:int(round(len(y_1)/2))]
          fst_part_y2 = y_2[:int((len(y_2)/2))]
          snd_part_y1 = y_1[int(round(len(y_1)/2)):]
          snd_part_y2 = y_2[int(len(y_2)/2):]
         
        y_train = np.array(fst_part_y1 + fst_part_y2)
        print("len y_train: ", len(y_train))
        y_val = np.array(snd_part_y1 + snd_part_y2)
        print("len y_val: ", len(y_val))
        
        y_train = np.array(fst_part_y1 + fst_part_y2)
        y_val = np.array(snd_part_y1 + snd_part_y2)
        
        print("y_train: ", y_train)
        print("len y_train: ", len(y_train))
        print("y_val: ", y_val)
        print("len y_val: ", len(y_val))
        
        X_train = X[:len(y_train)] #1
        print("len X_train: ", len(X_train))
        X_val = X[len(y_val):] #0
        print("len X_val: ", len(X_val))  
        
        '''n = len(X)
        X_train = X[:n//2]
        X_val = X[n//2:]
        y_train = y[:n//2]#y[0]
        y_val = y[n//2:]#y[1:n]'''
        #feature_map = kernel.to_qiskit_circuit()
        model = OneClassQSVM(kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, pos_label=1, average='binary')
      
    def evaluate_test(self, kernel: Kernel, K:np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the current kernel and return the corresponding wrong labels.
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :return: wrong_labels
        """

        '''n = len(X)
        X_train = X[:n//2]
        X_val = X[n//2:]
        y_train = y[:n//2]#y[0]
        y_val = y[n//2:]#y[1:n]'''
        #feature_map = kernel.to_qiskit_circuit()
        print("len X: ", len(X))
        print("len y: ", len(y))
        print("y: " , y)
        y_1 = list(filter(lambda x: x == 1, y)) #1
        print("len y_1: ", len(y_1))
        y_2 = list(filter(lambda x: x == 0, y)) #0
        print("len y_2: ", len(y_2))
        print("y_1: ", y_1)
        print("y_2: ", y_2)
        
        if int(len(y_1)/2) % 2 == 0: 
          fst_part_y1 = y_1[:int(len(y_1)/2)]
          fst_part_y2 = y_2[:int(len(y_2)/2)]
          snd_part_y1 = y_1[int(len(y_1)/2):]
          snd_part_y2 = y_2[int(len(y_2)/2):]
        else:
          fst_part_y1 = y_1[:int(round(len(y_1)/2))]
          fst_part_y2 = y_2[:int((len(y_2)/2))]
          snd_part_y1 = y_1[int(round(len(y_1)/2)):]
          snd_part_y2 = y_2[int(len(y_2)/2):]
         
        y_train = np.array(fst_part_y1 + fst_part_y2)
        print("len y_train: ", len(y_train))
        y_val = np.array(snd_part_y1 + snd_part_y2)
        print("len y_val: ", len(y_val))
        
        y_train = np.array(fst_part_y1 + fst_part_y2)
        y_val = np.array(snd_part_y1 + snd_part_y2)
        
        print("y_train: ", y_train)
        print("len y_train: ", len(y_train))
        print("y_val: ", y_val)
        print("len y_val: ", len(y_val))
        
        X_train = X[:len(y_train)] #1
        print("len X_train: ", len(X_train))
        X_val = X[len(y_val):] #0
        print("len X_val: ", len(X_val)) 
        
        
        print("test_set", X_train[0].shape)
        model = OneClassQSVM(kernel)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        conf_matrix = confusion_matrix(y_val,y_pred, labels=[1,-1])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=[1, -1], yticklabels=[1, -1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Test Set')
        plt.savefig('quask/plot_img/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        
        return f1_score(y_val, y_pred, pos_label=1, average='binary')
