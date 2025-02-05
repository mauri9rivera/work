import torch
import gpytorch
import numpy as np
import time
import matplotlib.pyplot as plt
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean

# Define the additive structure kernel
class AdditiveGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        
        # First additive component (dim 0 and 1)
        self.kernel1 = ScaleKernel(RBFKernel(active_dims=[0, 1]))
        
        # Second additive component (dim 2, 3, and 4)
        self.kernel2 = ScaleKernel(RBFKernel(active_dims=[2, 3, 4]))
        
        self.mean_module = ConstantMean()
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel1(x) + self.kernel2(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Synthetic test function
def additive_test_function(X):
    term1 = -torch.sum((X[:, :2] - 0.5)**2, dim=1)
    term2 = -torch.sum((X[:, 2:] + 0.5)**2, dim=1)
    return term1 + term2 + 0.1 * torch.randn_like(term1)

# Bayesian Optimization loop using GPyTorch native optimization
def run_bo(model_class, n_iter=100):
    bounds = torch.tensor([[0.0] * 5, [1.0] * 5])
    X = torch.rand(10, 5)  # Initial 5D points
    y = additive_test_function(X)  # Ensure y is 1D
    
    times = []
    best_values = []
    
    for i in range(n_iter):
        # Fit the model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = model_class(X, y, likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Train the model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        start_time = time.time()
        training_iterations = 50
        for _ in range(training_iterations):
            optimizer.zero_grad()
            output = model(X)  # Call the model's forward method
            loss = -mll(output, y)  # y is 1D
            loss.backward()
            optimizer.step()
        fit_time = time.time() - start_time
        times.append(fit_time)
        
        # Get best value
        best_values.append(y.max().item())
        
        # Find next point to evaluate using the model's predictive distribution
        model.eval()
        likelihood.eval()
        
        # Generate candidate points
        test_x = torch.rand(1000, 5)  # Random search for simplicity
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))  # Call the model's forward method
            mean = observed_pred.mean
            stddev = observed_pred.stddev
        
        # Select the point with highest upper confidence bound
        beta = 2.0  # Exploration-exploitation trade-off
        ucb = mean + beta * stddev
        next_point = test_x[torch.argmax(ucb)]
        
        # Evaluate new point
        new_y = additive_test_function(next_point.unsqueeze(0))
        
        # Update data
        X = torch.cat([X, next_point.unsqueeze(0)])
        y = torch.cat([y, new_y])
    
    return np.array(best_values), np.array(times)

# Define a regular GP for comparison
class RegularGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=5))
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Run comparison
additive_results, additive_times = run_bo(AdditiveGP)
regular_results, regular_times = run_bo(RegularGP)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(additive_results, label='Additive GP')
plt.plot(regular_results, label='Regular GP')
plt.title("Best Value Found")
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Additive GP', 'Regular GP'], 
        [additive_times.mean(), regular_times.mean()])
plt.title("Average Training Time per Iteration")
plt.ylabel("Seconds")

plt.tight_layout()
plt.show()