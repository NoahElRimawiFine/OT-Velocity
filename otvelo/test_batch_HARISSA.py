# %matplotlib notebook
import numpy as np
import pandas as pd
import sys
import os

from .utils_Velo import solve_velocities, visualize_pca, solve_prior, OT_lagged_correlation
#from my_utils_Velo import solve_velocities, OT_lagged_correlation
import copy


from .utils import *
import scipy
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import sklearn
import torch

#from my_utils_Velo import *

from utils import *
import scipy
import matplotlib.pyplot as plt

import evaluation_metrics
from evaluation_metrics import evaluate_AUC, early_precision#,stability


def batch_test(example, seeds, eps_samp=1E-2, l1=1E-2, alpha=0.5, save=False, tuning=False, modality='Corr', penalty='EN'):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    counter = 0
    for s in seeds:

        # Read count data in:
        FileName = '../Data/HARISSA/'+example+'/Data/data_'+str(s)+'.txt'
        
        counts = np.loadtxt(FileName)
        counts[2::,:] = np.log( counts[2::,:] + 1)
        
        counts = counts[1::,1::]
        n = counts.shape[0]
        
        Nt = 10
        
        
        
        num_cells_per_step = int( counts.shape[1]/Nt )
        
        print(counts.shape)
        
        
        labels = np.zeros((1,num_cells_per_step*Nt))
        
        
        counts_all = [[0]]*Nt
        for i in range(Nt):
            counts_all[i] = torch.tensor( counts[:,i*num_cells_per_step:(i+1)*num_cells_per_step] ) 
            
       
            labels[0,i*num_cells_per_step:(i+1)*num_cells_per_step] = i
        group_labels = list(range(Nt))
        for i in range(Nt):
            group_labels[i] = str(group_labels[i])

        n = counts.shape[0]
        
        
        Ts_prior,_ = solve_prior(counts,counts, Nt, labels, eps_samp=eps_samp, alpha=alpha)
        
        velocities_all, velocities_all_signed = solve_velocities( counts_all, Ts_prior, order = 1,stimulation=True)

        
        
        velocities = np.zeros( (n, (Nt)*num_cells_per_step))
        velocities_signed = np.zeros( (n, (Nt)*num_cells_per_step))
        
        for i in range(Nt):
            
            velocities[:,i*num_cells_per_step:(i+1)*num_cells_per_step] = velocities_all[i]
            velocities_signed[:,i*num_cells_per_step:(i+1)*num_cells_per_step] = velocities_all_signed[i]
        
        # Load truth of graph
        FileTruth = '../Data/HARISSA/'+example+'/True/inter_'+str(s)+'.npy'
        truth = np.load(FileTruth)
        Tv_true = truth
        Tv_true = Tv_true - np.diag( np.diag(Tv_true) )
        Tv_true = np.abs( Tv_true )
        
        
        
        if modality == 'Granger':
           
            if penalty == 'LASSO':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True, elastic_Net=True,tune=False,l1_opt=1.0, signed=True )
            elif penalty == 'EN':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True, elastic_Net=True,tune=False,l1_opt=0.5, signed=True )
            elif penalty == 'Ridge':
                 Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True, elastic_Net=True,tune=False,l1_opt=0.0, signed=True )
            elif penalty == 'CV':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True, elastic_Net=True,tune=True, signed=True )
            else:
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True, elastic_Net=True,tune=False,l1_opt=eps_feat )
            
        else:
            Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=True )
        
            
        Tv_total = Tv_total - np.diag( np.diag( Tv_total ))
        
        if save:
            if modality == 'Granger':
                if penalty is not None:
                    PATH = '../Data/HARISSA/'+example+'/OTVelo-Granger'
                    if not os.path.exists(PATH):
                        
                        os.makedirs(PATH)
                    FileOurs = PATH+'/score_'+str(s)+'penalty'+penalty+'.npy'
                
                
            else:
                PATH = '../Data/HARISSA/'+example+'/OTVelo-Corr'
                if not os.path.exists(PATH):
                        
                    os.makedirs(PATH)
                FileOurs = PATH + '/score_'+str(s)+'.npy'
                if tuning:
                    FileOurs = PATH+'/score_'+str(s)+'_alpha_'+str(alpha)+'_eps_'+str(eps_samp)+'.npy'                 
            np.save(FileOurs, Tv_total)
        
        Tv_total = np.abs( Tv_total )
        AUPRC[counter], AUROC[counter], random = evaluate_AUC( Tv_total, Tv_true, sign = None )
        
        print('AUPRC='+str( AUPRC[counter] ) )
        
        print('AUROC='+str( AUROC[counter] ) )
        counter += 1
    return AUROC, AUPRC

def tune_regression( example, seeds, eps_samp=1E-2, alpha=0.5, save=False, lambdas = [0.1,0.4,0.7,1.0,1.3,1.6], l1_ratios = [0.,0.5,1.] ):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    counter = 0
    for s in seeds:

        # Read count data in:
        FileName = '../Data/HARISSA/'+example+'/Data/data_'+str(s)+'.txt'
        
        counts = np.loadtxt(FileName)
        counts[2::,:] = np.log( counts[2::,:] + 1)
        
        # counts = counts[2::,1::]
        counts = counts[1::,1::]
        n = counts.shape[0]
        
        Nt = 10
        
        
        
        num_cells_per_step = int( counts.shape[1]/Nt )
        
        print(counts.shape)
        
        
        labels = np.zeros((1,num_cells_per_step*Nt))
        
        
        counts_all = [[0]]*Nt
        for i in range(Nt):
            counts_all[i] = torch.tensor( counts[:,i*num_cells_per_step:(i+1)*num_cells_per_step] ) 
            
       
            labels[0,i*num_cells_per_step:(i+1)*num_cells_per_step] = i
        group_labels = list(range(Nt))
        for i in range(Nt):
            group_labels[i] = str(group_labels[i])

        n = counts.shape[0]
        counts_pca, pca = visualize_pca(counts[1::,:],labels,group_labels,viz_opt='pca')
        
        #print('alpha: '+str(alpha) )
        #print('eps: '+str(eps_samp) )
        Ts_prior,_ = solve_prior(counts,counts, Nt, labels, eps_samp=eps_samp, alpha=alpha)
        #print(Ts_prior)
        velocities_all, velocities_all_signed = solve_velocities( counts_all, Ts_prior, order = 1,stimulation=True)
        
        velocities = np.zeros( (n, (Nt)*num_cells_per_step))
        velocities_signed = np.zeros( (n, (Nt)*num_cells_per_step))
        
        for i in range(Nt):
            
            velocities[:,i*num_cells_per_step:(i+1)*num_cells_per_step] = velocities_all[i]
            velocities_signed[:,i*num_cells_per_step:(i+1)*num_cells_per_step] = velocities_all_signed[i]
        
        
        for l1 in l1_ratios:
            for a in lambdas:
                import copy
                velocities_all_signed_copy = copy.deepcopy( velocities_all_signed )
                velocities_signed_copy = copy.deepcopy( velocities_signed )
                
                Tv_total = OT_lagged_correlation(velocities_all_signed_copy, velocities_signed_copy, Ts_prior, stimulation=True, elastic_Net=True,tune=False,l1_opt=l1, alpha_opt=a,signed=True )
                FileOurs = '../Data/HARISSA/'+example+'/OTVelo-Granger/score_'+str(s)+'_lam_'+str(a)+'_l1_'+str(l1)+'.npy'                 
                np.save(FileOurs, Tv_total)




def load_others_result(example, method, seeds, eps_samp=None, alpha=None, eps_feat=None,  sign=None, penalty='l1', l1=None, lam=None, tune=False):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    EP = [0]*len(seeds)
    #print( len( AUPRC) )
    counter = 0
    
    
        
    
    
    for s in seeds:

        # Load ground truth
        
        if example in ['FN4','CN5','BN8','FN8']:
            # These datasets use the identical graph
            FileTruth =  '../Data/HARISSA/'+example+'/True/inter_signed.npy'
            Tv_true = np.load(FileTruth)
              
            Tv_true = Tv_true - np.diag( np.diag(Tv_true) )
        
        elif example[0:4] == 'Tree':
            FileTruth =  '../Data/HARISSA/'+example+'/True/inter_'+str(s)+'.npy'
            Tv_true = np.load(FileTruth)
           
            Tv_true = Tv_true - np.diag( np.diag(Tv_true) )

        # Load resulting weight matrix
        
        if eps_samp == None or alpha == None:
            if method != 'OTVelo-Granger' and method != 'OTVelo-CV':
                FileResult = '../Data/HARISSA/'+example+'/'+method+'/score_'+str(s)+'.npy'
            else:
                # This requires specifying the regularization for regression step
                
                FileResult = '../Data/HARISSA/'+example+'/'+method+'/score'+'_'+str(s)+'penalty'+penalty+'.npy'
                
        elif tune == True and method == 'OTVelo-Corr':
            FileResult = '../Data/HARISSA/'+example+'/'+method+'/score_'+str(s)+'_alpha_'+str(alpha)+'_eps_'+str(eps_samp)+'.npy'
        
        if tune == True and method == 'OTVelo-Granger':
            FileResult = '../Data/HARISSA/'+example+'/OTVelo-Granger/score_'+str(s)+'_lam_'+str(lam)+'_l1_'+str(l1)+'.npy' 
            
        
        #print(FileResult)
        Tv_total = np.load(FileResult)

        # This is what CARDAMOM script does
        
        # if method == 'GENIE3':
        #     Tv_total = Tv_total.T
        
        
        Tv_total = Tv_total - np.diag( np.diag( Tv_total ))
        
        n = Tv_total.shape[0]
        Tv_total_flattened = []
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_total_flattened += [Tv_total[i,j]]
        Tv_total_flattened = np.array( Tv_total_flattened )

        
        
        
        
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score

        AUPRC[counter], AUROC[counter], random = evaluate_AUC( Tv_total, Tv_true, sign = sign )
        EP[counter] = early_precision( Tv_total, Tv_true, sign=sign)
        
        counter += 1
        
    return AUPRC, AUROC, EP, random




