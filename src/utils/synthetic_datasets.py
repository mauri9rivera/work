import random
import numpy as np
from botorch.test_functions.synthetic import Hartmann, Ackley, Griewank, Michalewicz
import torch

class SyntheticTestFun:

    def __init__(self, name, d, noise, negate):
        """Base constructor for synthetic test functions.

        Arguments:
            name: Name of BoTorch's test_functions among Hartmann, Ackley, Griewank, Michalewicz
            noise_std: Standard deviation of the observation noise.
        """
        self.d = d
        match name:
            case 'hartmann':
                if d != 6:
                    raise ValueError("The HartMann function needs to be 6 dimensional")
                self.f = Hartmann(noise_std=noise, negate=negate)
            case 'ackley':
                self.f = Ackley(d, noise_std=noise, negate=negate)
            case 'grienwank':
                self.f = Griewank(d, noise_std=noise, negate=negate)
            case 'michaelwicz':
                self.f = Michalewicz(d, noise_std=noise, negate=negate)
        self.lower_bounds = np.array(self.f._bounds)[:, 0]
        self.upper_bounds = np.array(self.f._bounds)[:, 1]


    def simulate(self, n_samples):
        """
        Simulate n_samples number of function calls to the test function.
        
        Returns: (X, Y) tuple of length n_samples containing those simulations.
        """
        X = torch.tensor(self.lower_bounds + (self.upper_bounds - self.lower_bounds), dtype=float) * torch.rand(n_samples, self.lower_bounds.shape[0])

        return X, self.f.forward(X)


