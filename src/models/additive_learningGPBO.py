import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.gridspec import GridSpec
import sys
import os
from pathlib import Path
import torch
import gpytorch
import math
import datetime
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import itertools
from mpl_toolkits.mplot3d import Axes3D
import time
import unittest
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from functools import reduce



### MODULES HANDLING ###
sys.path.append(str(Path('./').resolve().parent.parent))
from model_utils import optimize
from gaussians import GP, AdditiveLearningGP

### GLOBAL VARIABLES ###
DEVICE = 'cuda' #'cuda'
FILE_PATH = './data/5d_rats_dataset/5D_step4.mat'

### DATA HANDLING FUNCTIONS ###
def create_mean_map(resp):

    resp_mu = np.mean(resp,axis=0)
    mean_map = np.zeros((4,4,4,4,8,4))

    for e in range(resp.shape[1]):

        val_pw = np.unique(param[:,0])
        val_freq = np.unique(param[:,1])
        val_duration = np.unique(param[:,2])
        val_pos = np.unique(param[:,4])

        for i in range(len(param)):

            idx_pw = np.where(np.isclose(val_pw, param[i, 0]))[0][0]
            idx_freq = np.where(np.isclose(val_freq, param[i, 1]))[0][0]
            idx_duration = np.where(np.isclose(val_duration, param[i, 2]))[0][0]
            #idx_ch = int(np.where(np.isclose(val_ch,param[i, 4]))[0][0]-1)
            idx_ch = int(param[i, 4]-1)

            x_ch = ch2xy[idx_ch,0]
            y_ch = ch2xy[idx_ch,1]

            mean_map[e, idx_pw, idx_freq, idx_duration, x_ch, y_ch] = resp_mu[e,i]

    return mean_map


### Kernel Learning Functions ###
def sample_categorical(prob_partition, size):
    #e.g. sample_cateogircal([0.1, 0.1, 0.4, 0.2, 0.2], 4) = array([2, 2, 2, 3])
    return np.random.choice(len(prob_partition), size=size, p=prob_partition)

def sample_struct_priors(xx, yy, fixhyp):
    dx = xx.shape[1]
    n_partition = dx

    hyp = {}
    #model = SingleTaskGP(xx, yy, outcome_transform=Standardize(m=1))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    xx = xx.to(DEVICE)
    yy = yy.to(DEVICE)
    likelihood = likelihood.to(DEVICE)
    model = GP(xx, yy, likelihood, gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims= n_partition)))
    model = model.to(DEVICE)
    

    if all(k in fixhyp for k in ["l", "sigma", "sigma0"]):
        hyp["l"] = fixhyp["l"]
        hyp["sigma"] = fixhyp["sigma"]
        hyp["sigma0"] = fixhyp["sigma0"]
        decomp = learn_partition(xx, yy, hyp, fixhyp, n_partition, model, likelihood)
    else:
        prob_partition = np.ones(n_partition) / n_partition
        decomp = fixhyp.get("z", sample_categorical(prob_partition, dx))

        # GPytorch Model Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 50
        model.train()
        likelihood.train()

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets).mean()
            loss.backward()
            optimizer.step()

        # Extract optimized hyperparameters
        hyp["l"] = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze()

        hyp["sigma"] = model.covar_module.outputscale.detach().cpu().numpy()
        hyp["sigma0"] = model.likelihood.noise.detach().cpu().numpy()

        decomp = learn_partition(xx, yy, hyp, fixhyp, n_partition, model, likelihood)
        hyp["z"] = decomp
        

    return decomp, hyp

def learn_partition(xx, yy, hyp, fixhyp, n_partition, model, likf):
    if "decomp" in fixhyp:
        return fixhyp["decomp"]

    N_gibbs = 100
    gibbs_iter = N_gibbs // 2
    dim_limit = 3
    maxNdata = 750

    Nidx = min(maxNdata, xx.shape[0])
    xx = xx[:Nidx]
    yy = yy[:Nidx]

    hyp_dirichlet = np.ones(n_partition)
    prob_partition = hyp_dirichlet / hyp_dirichlet.sum()

    z = fixhyp.get("z", sample_categorical(prob_partition, xx.shape[1]))

    z_best = z.copy()
    minnlz = float('inf')

    for i in range(N_gibbs):
        for d in range(xx.shape[1]):
            log_prob = np.full(n_partition, -np.inf)
            nlz = np.full(n_partition, float('inf'))

            for a in range(n_partition):
                z[d] = a

                if i >= gibbs_iter and np.sum(z == a) >= dim_limit:
                    continue

                mll = ExactMarginalLogLikelihood(likf, model)
                model.train()
                nlz[a] = -mll(model(model.train_inputs[0]), model.train_targets).item()
                log_prob[a] = np.log(np.sum(z == a) + hyp_dirichlet[a]) - nlz[a]

            z[d] = np.argmax(log_prob - np.log(-np.log(np.random.rand(n_partition))))

            if minnlz > nlz[z[d]]:
                z_best = z.copy()
                minnlz = nlz[z[d]]

    return z_best

def partition_helper(arr):
    result = [[i for i, val in enumerate(arr) if val == y] for y in arr]
    result = [sublist for sublist in result if sublist]
    unique_lst = []
    for sublist in result:
        if sublist not in unique_lst:
            unique_lst.append(sublist)

    return unique_lst

def lengthscales_helper(model, old: list):

    ordered_lengthscales, ordered_outputscales = [], []

    for dim in range(model.n_dims):

        for idx, sublist in enumerate(old):

            if dim in sublist:
                
                ordered_lengthscales.append(model.kernels[idx].base_kernel.lengthscale[0]) #TODO: affected by ard_num_dims model.covar_module.base_kernel.lengthscale[idx][dim]
                ordered_outputscales.append(model.kernels[idx].outputscale)

    return ordered_lengthscales, ordered_outputscales


### GPBO ###
def additive_learningGPBO(GP_model, this_opt, options=None):

    # Setting default options
    if options is None:
        options = {}
    options.setdefault('n_subjects', 4)
    options.setdefault('n_dims', 5)
    options.setdefault('dim_sizes', np.array([8,4,4,4,4]))
    options.setdefault('which_opt','kappa') 
    options.setdefault('rho_low', 0.1)
    options.setdefault('rho_high', 6.0) 
    options.setdefault('nrnd', 5)
    options.setdefault('noise_min', 0.25)
    options.setdefault('noise_max', 10)
    options.setdefault('MaxQueries', 200) 
    options.setdefault('kappa', 5.0) 
    options.setdefault('nRep', 30) 

    #Params initialization
    n_subjects = options['n_subjects']
    n_dims = options['n_dims']
    DimSearchSpace = np.prod(options['dim_sizes'])
    rho_low = options['rho_low']
    rho_high = options['rho_high']
    nrnd = options['nrnd']
    noise_min = options['noise_min']
    noise_max = options['noise_max']
    MaxQueries = options['MaxQueries']
    kappa = options['kappa']
    nRep = options['nRep']
    total_size = np.prod(options['dim_sizes'])

    #Metrics initialization
    PP = torch.zeros((n_subjects,1,len(this_opt),nRep, MaxQueries), device=DEVICE)
    PP_t = torch.zeros((n_subjects,1, len(this_opt),nRep, MaxQueries), device=DEVICE)
    LOSS = torch.zeros((n_subjects,1, len(this_opt),nRep, MaxQueries), device=DEVICE)
    Q = torch.zeros((n_subjects,1,len(this_opt),nRep, MaxQueries), device=DEVICE)
    Train_time = torch.zeros((n_subjects,1, len(this_opt),nRep, MaxQueries), device=DEVICE)

    for s_idx in range(n_subjects):

        # "Ground truth" map
        MPm= torch.mean(peak_resp[:, s_idx], axis = 0)
        mMPm= torch.max(MPm)

        kappa= this_opt[0]

        # Create kernel, likelihood and priors
        # Put a  prior on the two lengthscale hyperparameters, the variance and the noise
        #The lengthscale parameter is parameterized on a log scale to constrain it to be positive
        #The outputscale parameter is parameterized on a log scale to constrain it to be positive
        lengthscale_prior= gpytorch.priors.SmoothedBoxPrior(a=math.log(rho_low),b= math.log(rho_high), sigma=0.01) 
        prior_lik= gpytorch.priors.SmoothedBoxPrior(a=noise_min**2,b= noise_max**2, sigma=0.01) # gaussian noise variance
        likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
        likf.noise= [1.0] 

        #Initialize subject metrics
        perf_explore= torch.zeros((nRep, MaxQueries), device=DEVICE)
        perf_exploit= torch.zeros((nRep, MaxQueries), device=DEVICE)
        loss = torch.zeros((nRep, MaxQueries), device=DEVICE)
        perf_rsq= torch.zeros((nRep), device=DEVICE)
        train_time = torch.zeros((nRep, MaxQueries), device=DEVICE)
        P_test =  torch.zeros((nRep, MaxQueries, 2), device=DEVICE)
        P_max_all_temp= torch.zeros((nRep, MaxQueries), device=DEVICE)

        for rep_i in range(nRep): # for each repetition

            if rep_i % 10 == 0:
                print('rep: ' + str(rep_i))
            
            MaxSeenResp=0 
            q=0 # query number                                
            order_this= np.random.permutation(DimSearchSpace) # random permutation of each entry of the search space
            P_max=[]
            hyp=[1.0]*(n_dims+2)

            while q < MaxQueries:

                # Burnout queries
                if q < nrnd:
                    P_test[rep_i][q][0]=int(order_this[q])
                
                #Find next point (max of acquisition function)
                else:
            
                    if torch.isnan(MapPrediction).any():
                        print('nan in Mean map pred')
                        MapPrediction = torch.nan_to_num(MapPrediction)

                    AcquisitionMap = MapPrediction + kappa*torch.nan_to_num(torch.sqrt(VarianceMap)) # UCB acquisition
                    NextQuery= torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
                    NextQuery = NextQuery[0][np.random.randint(len(NextQuery[0]))] if len(NextQuery[0]) > 1 else NextQuery[0][0] # Multiple maximums case
                    P_test[rep_i][q][0]= NextQuery

                query_elec = P_test[rep_i][q][0]
                sample_resp = peak_resp[:, s_idx, int(query_elec.item())]
                test_respo = sample_resp[np.random.randint(len(sample_resp))]
                test_respo += torch.normal(0.0, 0.02*torch.mean(sample_resp).item(), size=(1,), device=DEVICE).item() # size=(1,)                   
                
                #Edge cases handling
                if test_respo < 0:
                    test_respo=torch.tensor([0.0001], device= DEVICE)
                if test_respo==0 and q==0: # to avoid division by 0
                    test_respo= torch.tensor([0.0001], device=DEVICE)

                
                # done reading response
                P_test[rep_i][q][1]= test_respo # The first element of P_test is the selected search space point, the second the resulting value
                y=(P_test[rep_i][:q+1,1]) 


                # updated maximum response obtained in this round
                if (torch.max(torch.abs(y)) > MaxSeenResp) or (MaxSeenResp==0):
                    MaxSeenResp=torch.max(torch.abs(y))

                # search space position
                x= ch2xy[P_test[rep_i][:q+1,0].long(),:].float() 
                x = x.reshape((len(x),n_dims))

                y=y/MaxSeenResp
                y=y.float()
                
                # Initialization of the model and the constraint of the Gaussian noise 
                if q==0:
                    kernel_hyps = {}
                    z, kernel_hyps = sample_struct_priors(x, y, kernel_hyps)
                    partition = partition_helper(z)
                    print(f'Iter {q} - Decomposition of inputs: {partition}')
                    likf.noise= hyp[n_dims+1]
                    model = GP_model(x, y, likf, lengthscale_prior, partition) 
                    print(f'The model was properly initialized')
                    if DEVICE=='cuda':
                        model=model.cuda()
                        likf=likf.cuda()
                elif q % 20 == 0:
                    z, hyp = sample_struct_priors(x, y, kernel_hyps)
                    old_partition = partition
                    partition = partition_helper(z)
                    print(f'Iter {q} - Decomposition of inputs: {partition}')
                    ordered_lengthscales, ordered_outputscales = lengthscales_helper(model, old_partition)
                    print(f'Here is the order of the kernels ordered: {ordered_lengthscales}')
                    model = GP_model(x, y, likf, ordered_lengthscales, partition) 
                    print(f'The model was properly initialized for update')
                    if DEVICE=='cuda':
                        model=model.cuda()
                        likf=likf.cuda()
                else:
                    model.set_train_data(x,y, strict=False)

                start_time = time.time()
                #Training and optimizing model
                model.train()
                likf.train()
                print(f'device of model: {model.device}')
                print(f'device of one kernel? {model.kernels[0].device}')
                model, likf, l = optimize(model, likf, 10, x, y, verbose= False)

                train_time[rep_i, q] = time.time() - start_time

                #Evaluating model
                model.eval()
                likf.eval()

                with torch.no_grad():
                    X_test= ch2xy  
                    observed_pred = likf(model(X_test))

                VarianceMap= observed_pred.variance
                MapPrediction= observed_pred.mean

                # We only test for gp predictions at electrodes that we had queried (presumable we only want to return an electrode that we have already queried). 
                Tested= torch.unique(P_test[rep_i][:q+1,0]).long()
                MapPredictionTested=MapPrediction[Tested]
                if len(Tested)==1:
                        BestQuery=Tested
                else:
                    BestQuery= Tested[(MapPredictionTested==torch.max(MapPredictionTested)).reshape(len(MapPredictionTested))]
                    if len(BestQuery) > 1:  
                        BestQuery = np.array([BestQuery[np.random.randint(len(BestQuery))].cpu()])

                # Maximum response at time q 
                P_max.append(BestQuery.item())
                loss[rep_i, q] = l
                q+=1

            # estimate current exploration performance: knowledge of best stimulation point    
            perf_explore[rep_i,:]=MPm[P_max].reshape((len(MPm[P_max])))/mMPm
            # estimate current exploitation performance: knowledge of best stimulation point 
            perf_exploit[rep_i,:]= P_test[rep_i][:,0].long()

        PP[s_idx,0,0]=perf_explore
        Q[s_idx,0,0] = P_test[:,:,0]
        PP_t[s_idx,0,0]= MPm[perf_exploit.long().cpu()]/mMPm
        Train_time[s_idx, 0, 0] = train_time
        LOSS[s_idx,0,0] = loss

    np.savez('./output/experiments/'+ model.name+'_'+datetime.date.today().strftime("%y%m%d")+'_4channels_artRej_lr001_5rnd.npz', PP=PP.cpu(), PP_t=PP_t.cpu(), LOSS= LOSS.detach().cpu().numpy(), Train_time=Train_time.detach().cpu().numpy(), Q = Q.cpu(), this_opt = this_opt, nrnd = nrnd, kappa = kappa)


if __name__ == '__main__':

    # Note: currently setup for peak_map response
    data = scipy.io.loadmat(FILE_PATH)['stim_combinations'] #scipy.io.loadmat(file_path)['Data']
    resp = scipy.io.loadmat(FILE_PATH)['emg_response'] #data[0][0][0]
    param = scipy.io.loadmat(FILE_PATH)['stim_combinations']# data[0][0][1]
    ch2xy = param[:32,[-2,-1]].astype(int) -1
    peak_resp = resp[:, :, :, 0]
    peak_map = create_mean_map(peak_resp)
    peak_resp = torch.from_numpy(peak_resp).float().to(DEVICE)
    ch2xy = param[:, [0,1,2,5,6]]
    ch2xy = torch.from_numpy(ch2xy).float().to(DEVICE)

    options = {} # Here's where you change the hyperparams and whatnot
    options['kappa'] = 5.0
    additive_learningGPBO(AdditiveLearningGP, np.array([5.0])) # temporospatialGPBO(ParallelizedGP, np.array([5.0]))