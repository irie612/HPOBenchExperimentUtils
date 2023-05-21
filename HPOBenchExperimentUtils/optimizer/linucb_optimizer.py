from pathlib import Path
from typing import Dict, Union
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
from HPOBenchExperimentUtils.optimizer.base_optimizer import Optimizer
import numpy as np

class LinUCBOptimizer(Optimizer):
    def __init__(self, benchmark: Bookkeeper, settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        self.alpha = settings['alpha']
        self.num_samples = settings['num_samples'] # K in the literature - num arms/num samples
        self.num_hyperparameters = len(self.cs.get_hyperparameters()) # d in the literature - feature space
        

    def setup(self):
        pass

    def run(self):
        # sample num_samples hyperparameter configurations from the configuration space
        # each configuration is a d-dimensional vector
        hyperparameter_configurations = self.cs.sample_configuration(size=self.num_samples) # not sure if this is right look at the ConfigSpace docs
        
        # init array A of shape (K x d x d) where each A[i] is a d x d identity matrix
        A = np.stack((np.identity(self.num_hyperparameters),)*self.num_samples, axis=0)
        # init array b of shape (K x d) where each b[i] is a d-dimensional zero vector
        b = np.zeros((self.num_samples, self.num_hyperparameters))
        # init timestep counter
        t = 0
        # repeat until true as the benchmark will stop the run
        while True:
            t += 1
            
            for hc in hyperparameter_configurations:
                # compute theta_hat
                theta_hat = np.linalg.inv(A) @ b
                break
            break
        print(self.cs)
        print(f"alpha: {self.alpha}")
        print(f"num_samples: {self.num_samples}")
        print(f"num_hyperparameters: {self.num_hyperparameters}")