import os
import torch
import gpytorch

def save_model(GP_model, model_file_name='change_this_model_name.pth'):

    file_path = os.path.join('../models', model_file_name)

    torch.save(GP_model.state_dict(), file_path)

    print(f"Model {model_file_name} successfully saved.")
    

def load_model(GP_model, model_file_name):
    #TODO: Will this required adding extra args if there's more specifications to the precise GP model to build?

    state_dict = torch.loard(f'../models/{model_file_name}')

    model = GP_model()

    model.load_state_dict(state_dict)

    return model

def optimize(model, likelihood, training_iter, train_x, train_y, verbose= False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters lr= 0.01
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    l = 0.0

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)

      
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        l += loss
        loss.backward()

        if verbose== True:

            print('Iter %d/%d - Loss: %.3f   lengthscale_1: %.3f   lengthscale_2: %.3f   lengthscale_3: %.3f   lengthscale_4: %.3f    lengthscale_4: %.3f    kernelVar: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale[0][0].item(),
            model.covar_module.base_kernel.lengthscale[0][1].item(),
            model.covar_module.base_kernel.lengthscale[0][2].item(),
            model.covar_module.base_kernel.lengthscale[0][3].item(),
            model.covar_module.base_kernel.lengthscale[0][4].item(),
            model.covar_module.outputscale.item(),
            model.likelihood.noise.item()))

        optimizer.step()

    return model, likelihood, l / training_iter
