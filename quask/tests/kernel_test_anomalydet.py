from quask.utils.utils_data import *
import warnings
from quask.optimizer import GenOptimizer
from quask.evaluator import OneClassQSVMEvaluator
from quask.core import Ansatz, KernelFactory, KernelType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")   

def compute_statistics(f1_scores: list):
    #Statistics

    total_mean = np.mean(f1_scores)
    total_dev_std = np.std(f1_scores)
    total_var = np.var(f1_scores)

    statistics_dataframe = pd.DataFrame(
        {
            'Metrics': ['total_mean' , 'total_devstd' , 'total_var'],
            'Value: ': [total_mean, total_dev_std, total_var]
        }
        
    )

    statistics_dataframe.to_excel('./quask/plot_img/stats.xlsx', sheet_name='Stats', na_rep='NaN', float_format='%.4f', header=True, index=True, index_label='Statistic', startrow=0,
    startcol=0, engine='openpyxl', freeze_panes=(1, 0))

    plt.hist(f1_scores,bins=10, edgecolor='blue')
    plt.axvline(total_mean, color='red', linestyle='dashed', linewidth=1, label=f'Total Mean: {total_mean:.2f}')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Value')
    plt.savefig('./quask/plot_img/confusion_matrix_test.png', dpi=300, bbox_inches='tight')

def test_0(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    N_FEATURES = len(X_train[0]) # numero di feature del dataset
    N_OPERATIONS = 15 # 
    N_QUBITS = N_FEATURES + 2
    print("N_FEATURES",N_FEATURES)
    f1_scores = []    
    max_fitn_kern = 0
    init_ansatz = Ansatz(n_features=N_FEATURES, n_qubits=N_QUBITS, n_operations=N_OPERATIONS)
    init_ansatz.initialize_to_identity()
    init_kernel = KernelFactory.create_qiskit_kernel(init_ansatz, "Z" * N_QUBITS, KernelType.FIDELITY)
    qsvm_evaluator = OneClassQSVMEvaluator({ 'nqubits' : 4, 'feature_map': ' ', 'nu_param' : 0.01})
    genetic_optimizer = GenOptimizer(init_kernel, X_train, y_train, qsvm_evaluator)

    for i in range(10):
        optim_kernel = genetic_optimizer.optimize() 
        f1 = qsvm_evaluator.evaluate(optim_kernel, None, X_train, y_train)
        f1_scores.append(f1)
        
        if abs(max_fitn_kern) < max(abs(max_fitn_kern), abs(f1)):
            max_fitn_kern = max(abs(max_fitn_kern), abs(f1))
            best_optim_kernel = optim_kernel

    kernel_on_test_set = qsvm_evaluator.evaluate_test(best_optim_kernel, None, X_test, y_test)
    
    print("KERNEL_ON_TEST_SET: ", kernel_on_test_set)
    compute_statistics(f1_scores)
    
    

if __name__ == "__main__" :
     
    #EVALUATE ON TRAINING
    args = {    'sig_path': './latentrep_AtoHZ_to_ZZZ_35.h5', 
                    'bkg_path': './latentrep_QCD_sig.h5',
                    'test_bkg_path': './latentrep_QCD_sig_testclustering.h5',
                    'unsup': False,
                    'nqubits' : 10,
                    'feature_map': 'u_dense_encoding',
                    'run_type' : 'ideal',
                    'output_folder' : 'quantum_test',
                    'nu_param' : 0.01,
                    'ntrain':  150,
                    'quantum': True,
                    'ntest': 0
            }

    #print(f'X: {X}')
    #print(f'y: {y}')
    train_loader, test_loader = get_data(args)
    X_train, y_train = train_loader[0], train_loader[1]
    print(f'train_loader 0: {X_train}')
    print(f'train_loader 1: {y_train}')
    
    
    #EVALUATE ON TEST
    test_args = {'sig_path': './latentrep_RSGraviton_WW_BR_15.h5', 
                'bkg_path': './latentrep_RSGraviton_WW_NA_35.h5',
                'test_bkg_path': './latentrep_QCD_sig_testclustering.h5',
                'unsup': False,
                'nqubits' : 10,
                'feature_map': 'u_dense_encoding',
                'run_type' : 'ideal',
                'nu_param' : 0.01,
                'ntrain':0,
                'kfolds': 5,
                'quantum': True,
                'ntest': 150
                }

    _, test_loader = get_data(test_args)
    X_test, y_test = test_loader[0], test_loader[1]
    print(f'test_loader 0: {X_test}')
    print(f'test_loader 1: {y_test}')
    
    #test 
    test_0(X_train, X_test, y_train, y_test)
    
    
    
       
