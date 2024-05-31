import numpy as np
import copy
from ..core import Operation, Ansatz, Kernel, KernelFactory
from ..evaluator import KernelEvaluator
from . import BaseKernelOptimizer, WideKernelEnvironment
from mushroom_rl.algorithms.value import QLearning, QLambda


class ReinforcementLearningOptimizer(BaseKernelOptimizer):
    """
    Reinforcement learning based technique for optimize a kernel function
    """

    def __init__(self, initial_kernel: Kernel, X: np.ndarray, y: np.ndarray, ke: KernelEvaluator):
        """
        Initialization
        :param initial_kernel: initial kernel object
        :param X: datapoints
        :param y: labels
        :param ke: kernel evaluator object
        """
        from mushroom_rl.core import Environment
        from mushroom_rl.core import Core
        from mushroom_rl.algorithms.value import SARSALambda
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.utils.parameters import Parameter
        from mushroom_rl.utils.dataset import compute_J
        self.initial_kernel = copy.deepcopy(initial_kernel)
        self.X = X
        self.y = y
        self.ke = ke
        WideKernelEnvironment.register()
        self.mdp = WideKernelEnvironment.make('WideKernelEnvironment', initial_kernel=self.initial_kernel, X=X, y=y, ke=ke)
        self.agent = None
        self.core = None

    def optimize(self, initial_episodes=3, n_episodes=100, n_steps_per_fit=1, final_episodes=3, algorithm='SARSA'):
        """
        Optimization routine
        :param initial_episodes:
        :param n_steps:
        :param n_steps_per_fit:
        :param final_episodes:
        :return:
        """
        from mushroom_rl.core import Environment
        from mushroom_rl.core import Core
        from mushroom_rl.algorithms.value import SARSALambda
        from mushroom_rl.policy import EpsGreedy
        from mushroom_rl.utils.parameters import Parameter
        from mushroom_rl.utils.dataset import compute_J
        # Policy
        epsilon = Parameter(value=1.)
        pi = EpsGreedy(epsilon=epsilon)
        learning_rate = Parameter(.1)

        # Agent
        if(algorithm == 'SARSA'):
            self.agent = SARSALambda(self.mdp.info, pi,
                            learning_rate=learning_rate,
                            lambda_coeff=.9)
        elif (algorithm == 'QLEARNING'):
            self.agent = QLearning(self.mdp.info, pi,
                            learning_rate=learning_rate)
        elif  (algorithm == 'QLAMBDA'):
            self.agent = QLambda(self.mdp.info, pi,
                            learning_rate=learning_rate,
                            lambda_coeff=.9)
            

        # Reinforcement learning experiment
        self.core = Core(self.agent, self.mdp)

        # Visualize initial policy for 3 episodes
        dataset = self.core.evaluate(n_episodes=initial_episodes, render=True)

        # Print the average objective value before learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function before learning: {J}')

        # Train
        self.core.learn(n_episodes=n_episodes, n_steps_per_fit=n_steps_per_fit, render=True)

        # Visualize results for 3 episodes
        dataset = self.core.evaluate(n_episodes=final_episodes, render=True)

        # Print the average objective value after learning
        J = np.mean(compute_J(dataset, self.mdp.info.gamma))
        print(f'Objective function after learning: {J}')

        print( "mdp_state:" + str(self.mdp._state[1:]))
        print( "mdp_nfeatures:" + str(self.mdp.n_features))
        print( "mdp_nqubit:" + str(self.mdp.n_qubits))
        print( "mdp_noperations:" + str(self.mdp.n_operations))
        print( "mdp_measurement:" + str(self.mdp.allow_midcircuit_measurement))
        
        kernel = Kernel.from_numpy(self.mdp._state[1:], self.mdp.n_features, self.mdp.n_qubits, self.mdp.n_operations, self.mdp.allow_midcircuit_measurement)
        print(kernel)
        return kernel
