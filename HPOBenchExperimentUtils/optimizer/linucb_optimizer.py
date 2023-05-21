import logging
import json
from pathlib import Path
from typing import Dict, Union
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer, Optimizer
import numpy as np
import time

_log = logging.getLogger(__name__)

# not sure if this should be single fidelity or standard optimizer
class LinUCBOptimizer(Optimizer):
    def __init__(self, benchmark: Bookkeeper, settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        self.alpha = settings['alpha']
        self.num_samples = settings['num_samples'] # K in the literature - num arms/num samples
        self.num_hyperparameters = len(self.cs.get_hyperparameters()) # d in the literature - feature space
        

    def setup(self):
        pass

    def run(self):
        # sample num_samples hyperparameter configurations from the configuration space. Each configuration is a d-dimensional vector
        hyperparameter_configurations = self.cs.sample_configuration(size=self.num_samples)

        # init array A of shape (K x d x d) where each A[i] is a d x d identity matrix
        A = np.stack((np.identity(self.num_hyperparameters),)*self.num_samples, axis=0)
        # init array b of shape (K x d) where each b[i] is a d-dimensional zero vector
        b = np.zeros((self.num_samples, self.num_hyperparameters))

        # init results list
        results = []

        # init timestep counter
        t = 0
        # repeat until true as the benchmark will stop the run
        while True:
            # print(f'timestep: {t}')
            # init theta_hat of shape (K x d) where each theta_hat[i] is a d-dimensional zero vector
            theta_hat = np.zeros((self.num_samples, self.num_hyperparameters))
            # init ucb of shape (K x 1) where each ucb[i] is a scalar
            ucb = np.zeros((self.num_samples, 1))
            for i, hc in enumerate(hyperparameter_configurations):
                # compute theta_hat
                theta_hat[i] = np.linalg.inv(A[i]) @ b[i]
                # compute ucb
                ucb[i] = theta_hat[i].T @ hc.get_array() + self.alpha * np.sqrt(hc.get_array().T @ np.linalg.inv(A[i]) @ hc.get_array())
            # print(f'ucb: {ucb}')
            # select the hyperparameter configuration with the highest ucb with ties broken randomly and get its index in the array
            best_index = np.random.choice(np.flatnonzero(ucb == np.max(ucb)))
            # get the hyperparameter configuration with the highest ucb
            best_hc = hyperparameter_configurations[best_index]
            # evaluate the hyperparameter configuration with the highest ucb
            result = self.benchmark.objective_function(configuration_id= f"Config{best_index}", 
                                                       configuration=best_hc,
                                                       rng=self.rng,
                                                       **self.settings_for_sending,
                                                       )
            # append the result to the results list
            results.append((best_hc, result))
            # get the used resources
            resources = self.benchmark.resource_manager.get_used_resources()
            # get the reward from the results
            reward = result['function_value']
            # log the result
            _log.info(f'Config{best_index} - Result {reward:.4f} - '
                      f'Time Used: {resources.total_time_used_in_s:.2f}'
                      f'{self.benchmark.resource_manager.limits.time_limit_in_s}')
            
            # save the result
            self.__save_results({"num_evals": t,
                                 "best_config": best_hc.get_dictionary(),
                                 "timestamp": time.time(),
                                 "function_value": reward,
                                 "config_name": f"Config{best_index}",})

            # update A and b
            A[best_index] += best_hc.get_array() @ best_hc.get_array().T
            b[best_index] += reward * best_hc.get_array()
            # print(f'A: {A}')
            # print(f'b: {b}')

            # increment timestep counter
            t += 1

    def __save_results(self, entry):
        with open(self.output_dir / "trajectory.json", "a") as fh:
            fh.write(json.dumps(entry) + "\n")