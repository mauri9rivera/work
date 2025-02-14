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
from gaussians import GP

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

### GPBO ###
def neurostim_BO(GP_model, this_opt, options=None):

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
    Q = torch.zeros((n_subjects,1,len(this_opt),nRep, MaxQueries), device=DEVICE)
    LOSS = torch.zeros((n_subjects,1, len(this_opt),nRep, MaxQueries), device=DEVICE)
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
        priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(rho_low),b= math.log(rho_high), sigma=0.01) 
        priorbox2= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) # std
        matk= gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= n_dims, lengthscale_prior= priorbox)

        matk_scaled = gpytorch.kernels.ScaleKernel(matk, outputscale_prior= priorbox2)
        matk_scaled.base_kernel.lengthscale= [1.0]*n_dims
        matk_scaled.outputscale= [1.0]
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
                    matk_scaled.base_kernel.lengthscale= hyp[:n_dims] # Update the initial value of the parameters 
                    matk_scaled.outputscale= hyp[n_dims]
                    likf.noise= hyp[n_dims+1]
                    m = m= GP_model(x, y, likf, matk_scaled)#GP_model(x, y, likf, [priorbox, priorbox2], 3, 2)#m= GP_model(x, y, likf, matk_scaled)
                    if DEVICE=='cuda':
                        m=m.cuda()
                        likf=likf.cuda()    
                # Update training data     
                else:
                    m.set_train_data(x,y, strict=False)

                start_time = time.time()
                #Training and optimizing model
                m.train()
                likf.train()
                m, likf, l = optimize(m, likf, 10, x, y, verbose= False)

                train_time[rep_i, q] = time.time() - start_time

                #Evaluating model
                m.eval()
                likf.eval()

                with torch.no_grad():
                    X_test= ch2xy  
                    observed_pred = likf(m(X_test))

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
                '''
                hyp= torch.tensor([m.covar_module.base_kernel.lengthscale[0][0].item(),
                                m.covar_module.base_kernel.lengthscale[0][1].item(),
                                m.covar_module.base_kernel.lengthscale[0][2].item(),
                                m.covar_module.base_kernel.lengthscale[0][3].item(),
                                m.covar_module.base_kernel.lengthscale[0][4].item(),
                                m.covar_module.outputscale.item(),
                                m.likelihood.noise[0].item()], device=DEVICE)
                '''
                #hyperparams[s_i, c_i, k_i,rep_i,q,:] = hyp    0
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
       

    np.savez('./output/experiments/'+ m.name+'_'+datetime.date.today().strftime("%y%m%d")+'_4channels_artRej_lr001_5rnd.npz', PP=PP.cpu(), PP_t=PP_t.cpu(), LOSS= LOSS.detach().cpu().numpy(), Train_time=Train_time.detach().cpu().numpy(), Q = Q.cpu(), this_opt = this_opt, nrnd = nrnd, kappa = kappa, options=options)

    return m

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
    neurostim_BO(GP, np.array([5.0]))