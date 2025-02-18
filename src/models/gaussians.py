import gpytorch
import math
import torch

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.name = 'VanillaGPBO'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TemporoSpatialGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, 
                ard_num_dims_group1=3, ard_num_dims_group2=2):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel( 
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group1, 
                                active_dims=[0, 1, 2],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior= scale_prior)
        self.temporal_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group1
        self.temporal_kernel.outputscale=[1.0]

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group2, 
                                active_dims=[3, 4],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior=scale_prior)
        self.spatial_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group2
        self.spatial_kernel.outputscale=[1.0]

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'TemporoSpatialGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.temporal_kernel(x) + self.spatial_kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class ParallelizedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, 
                ard_num_dims_group1=3, ard_num_dims_group2=2):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)
        
        self.temporal_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel( 
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group1, 
                                active_dims=[0, 1, 2],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior= scale_prior)
        self.temporal_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group1
        self.temporal_kernel.outputscale=[1.0]

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.MaternKernel(
                                mu=2.5, 
                                ard_num_dims=ard_num_dims_group2, 
                                active_dims=[3, 4],
                                lengthscale_prior= lengthscale_prior),
                                outputscale_prior=scale_prior)
        self.spatial_kernel.base_kernel.lengthscale = [1.0]*ard_num_dims_group2
        self.spatial_kernel.outputscale=[1.0]

        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'ParallelizedGP'

    def forward(self, x):
        mean_x = self.mean_module(x)

        # Parallelize kernel computations
        kernels = [self.temporal_kernel, self.spatial_kernel]
        covars = torch.nn.parallel.parallel_apply(kernels, [x, x])
        
        # Sum the covariance matrices
        covar_x = covars[0] + covars[1]
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class AdditiveLearningGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior, 
                group_dims):
        super().__init__(train_x, train_y, likelihood)

        scale_prior = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01)

        self.group_dims = group_dims
        self.kernels = []
        for group in group_dims:

            kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(
                            mu=2.5,
                            ard_num_dims= len(group),
                            active_dims=group,
                            lengthscale_prior=lengthscale_prior),
                            outputscale_prior= scale_prior)
            kernel.base_kernel.lengthscale = [1.0] * len(group)
            kernel.outputscale = [1.0]
            self.kernels.append(kernel)


        self.mean_module = gpytorch.means.ConstantMean()
        self.likelihood = likelihood
        self.name = 'KernelLearningGP'

    def forward(self, x):
        mean_x = self.mean_module(x)

        print(f'device of input: {x.device}')
        for kernel in self.kernels:
            print(f'device of kernel: {kernel.device}')
        # Parallelize kernel computations
        covars = torch.nn.parallel.parallel_apply(self.kernels, [x for _ in self.kernels])
        
        # Sum the covariance matrices
        covar_x = sum(covars)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)    