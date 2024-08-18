# %matplotlib notebook
import numpy as np
import pandas as pd
import sys
import os
#sys.path.append('../')

#from coot_torch import *
#from my_utils_Velo import *
from utils_Velo import solve_velocities, visualize_pca, solve_prior, OT_lagged_correlation
import copy

#from agw_scootr import *
from utils import *
import scipy
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys
import sklearn

#from agw_scootr import *
from utils import *
import scipy
import matplotlib.pyplot as plt
import evaluation_metrics
from evaluation_metrics import evaluate_AUC, early_precision
from sklearn.manifold import TSNE



def construct_truth_BEELINE( FileName, gene_names, sign = None):
    
    # This file constructs ground truth graph
    
    import scipy.io
    n = len(gene_names)
    truth_tab = pd.read_csv(FileName)#'../BEELINE-data/inputs/Curated/mCAD/mCAD-2000-1/refNetwork.csv')
    truth = np.zeros( (n,n))
    #print(truth_tab)
    for i in range(n):
        for j in range(n):
            idx = np.where( np.array(truth_tab['Gene1'] == gene_names[i]) * np.array(truth_tab['Gene2'] == gene_names[j]) )[0]
            if len(idx) > 0:
                if truth_tab.iloc[idx[0],2] == '+':
                    truth[i,j] = 1
                else:
                    truth[i,j] = -1
    truth_signed = truth
    
    if sign == None:
        truth = truth_signed
    elif sign == 'pos':
        truth[truth < 0] = 0
        truth = abs(truth)
    elif sign == 'neg':
        truth[truth > 0] = 0
        # truth = abs(truth)
    return truth


        
 
        




def batch_test_BEELINE(example, seeds, modality='Corr',dropout=None, branch_no=-1, alpha=0.5, eps_samp=1E-2, eps_feat=1E-2, save=False, tuning=False, penalty='EN'):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    counter = 0

    # Number of time points determined according to BEELINE appendix
    
    if example == 'GSD':
        Nt = 6
        
    if example == 'HSC':
        Nt = 20
        
    if example == 'mCAD':
        
        Nt = 10
    if example == 'VSC':
        
        Nt = 5
    for s in seeds:

        # Read count data in:
        
        if dropout == None:
            if branch_no == -1:
                FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/ExpressionData.csv'
                FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/PseudoTime.csv'
            else:
                FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/ExpressionData-'+str(branch_no)+'.csv'
                FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/PseudoTime.csv'
            
        else:
            if branch_no == -1:
                FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/ExpressionData.csv'
                FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/PseudoTime.csv'
            else:
                FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/ExpressionData-'+str(branch_no)+'.csv'
                FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/PseudoTime.csv'
            
        
        
        data = pd.read_csv(FileName)
        cell_names_all = list( data.columns)
        cell_names_all = cell_names_all[1::]
        gene_names = list(data.iloc[:,0])
        
        data = np.array( data.iloc[:,1::])
        
        data = np.array( data, dtype = float)
        
        time = np.zeros( data.shape[1])
        
        t = pd.read_csv(FileTime)

        if branch_no < 0:
            branch_cell = []
            for i in range( data.shape[1] ):
                if cell_names_all[i][-3] == '_':
                    time[i] = int(cell_names_all[i][-2::] )
                else:
                    time[i] = int(cell_names_all[i][-3::] )
               
                branch_cell += [ np.where( np.isnan(np.array(t.iloc[i,1:],dtype=float))==False)[0][0] ]
            branch_cell = np.array( branch_cell )
        else:
            t = t.iloc[:,branch_no+1]
            t = np.array( t )
            
            time = t[ np.isnan(t) == False ] 
        
        
        tQuantiles = pd.qcut( time, q = Nt, duplicates ='drop')
        mid = [(a.left + a.right)/2 for a in tQuantiles]
        time = np.array( mid )

        time_pts = set( list(time) )
        time_pts = np.sort( list(time_pts) )
        dt = time_pts[1:] - time_pts[0:len(time_pts)-1]
        
        counts = data
        
        n = counts.shape[0]
        
        counts_all = [[0]]*Nt
        branch_cell_all = [ [0] ]*Nt
        
        
        labels = np.zeros((1,counts.shape[1]))
        
        
        nums = [0]*Nt
        
        for i in range(Nt):
            idx = np.where(time == time_pts[i])[0]
            counts_all[i] = counts[:,idx]
           
            if branch_no == -1:
                branch_cell_all[i] = branch_cell[ idx ]
            nums[i] = len(idx)
        #print(nums)
        s_cum = np.array( [0]+list( np.cumsum( nums)) )
        for i in range(Nt):
            labels[0,s_cum[i]:s_cum[i+1]] = i
        group_labels = list(range(Nt))
        for i in range(Nt):
            group_labels[i] = str(group_labels[i])
        counts = counts_all[0]
        for j in range(Nt-1):
            counts = np.concatenate( (counts, counts_all[j+1] ), axis=1 )

        # Find velocity

        n = counts.shape[0]
        counts_pca, pca = visualize_pca(counts,labels,group_labels,viz_opt='pca')

        
        
        Ts_prior,_ = solve_prior(counts,counts, Nt, labels, eps_samp=eps_samp, alpha=alpha)
       
    
        
        for i in range(Nt-1):
            Ts_prior[i] = Ts_prior[i]/Ts_prior[i].sum()
        
        velocities_all, velocities_all_signed = solve_velocities( counts_all, Ts_prior, dt=dt, order = 1)
        velocities = np.zeros( counts.shape )
        velocities_signed = np.zeros( counts.shape )
        
        for i in range(Nt):
            
            velocities[:,s_cum[i]:s_cum[i+1]] = velocities_all[i]
            velocities_signed[:,s_cum[i]:s_cum[i+1]] = velocities_all_signed[i]
    
        
        
        
        if modality == 'Granger':
            
           
            if penalty == 'LASSO':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=False, elastic_Net=True,tune=False,l1_opt=1.0, signed=True )
            elif penalty == 'EN':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=False, elastic_Net=True,tune=False,l1_opt=0.5, signed=True )
            elif penalty == 'Ridge':
                 Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=False, elastic_Net=True,tune=False,l1_opt=0.0, signed=True )
            elif penalty == 'CV':
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=False, elastic_Net=True,tune=True, signed=True )
            else:
                Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, stimulation=False, elastic_Net=True,tune=False,l1_opt=eps_feat )
                
            
        else:
            
            Tv_total = OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior)
            
        #Tv_total = Tv_total - np.diag( np.diag( Tv_total ))
        Tv_total[np.isnan(Tv_total)] = 0
       
        if save:
            if modality == 'Granger':
                if dropout == None:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/OTVelo-Granger'
                    
                else:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/OTVelo-Granger'
                # if not os.path.exists(FileOurs):
                        
                #     os.makedirs(FileOurs)
                
                
                FileOurs = FileOurs + '_penalty'+penalty
                
                    
            else:
                if dropout == None:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/OTVelo-Corr'
                else:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/OTVelo-Corr'
                if not os.path.exists(FileOurs):
                        
                    os.makedirs(FileOurs)
                    
                
                if tuning:
                    FileOurs = FileOurs + '_alpha_'+str(alpha)+'_epscell_'+str(eps_samp)
                
                
            
            if branch_no == -1:
                FileOurs = FileOurs + '.npy'
            else:
                FileOurs = FileOurs + '_'+str(branch_no)+'.npy'
            print(Tv_total)
            np.save(FileOurs, Tv_total)
       
        # Load truth of graph
        if dropout == None:
            FileTruth = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/refNetwork.csv'
        else:
            FileTruth = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/refNetwork.csv'
        
        Tv_true = construct_truth_BEELINE( FileTruth, gene_names)
        # print(Tv_true.shape)
        Tv_true = Tv_true - np.diag( np.diag(Tv_true) )

        AUPRC[counter], AUROC[counter], random = evaluate_AUC( Tv_total, Tv_true, sign = None )
        print('AUPRC='+str( AUPRC[counter] ) )
        
        print('AUROC='+str( AUROC[counter] ) )
        
        counter += 1
    return AUROC, AUPRC


def tune_regression( example, seeds, branch_no = -1, eps_samp=1E-2, eps_feat=1E-2, alpha=0.5, save=False, lambdas = [0.1,0.4,0.7,1.0,1.3,1.6], l1_ratios = [0.,0.5,1.] ):
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    counter = 0

    if example == 'GSD':
        Nt = 6
        
    if example == 'HSC':
        Nt = 20
        
    if example == 'mCAD':
        
        Nt = 10
    if example == 'VSC':
        
        Nt = 5
    for s in seeds:

        # Read count data in:
        
        if branch_no == -1:
            FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/ExpressionData.csv'
            FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/PseudoTime.csv'
        else:
            FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/ExpressionData-'+str(branch_no)+'.csv'
            FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/PseudoTime.csv'
            
        
        
        data = pd.read_csv(FileName)
        cell_names_all = list( data.columns)
        cell_names_all = cell_names_all[1::]
        gene_names = list(data.iloc[:,0])
        
        data = np.array( data.iloc[:,1::])
        
        data = np.array( data, dtype = float)
        
            #ptime = pd.read_csv(tname)
        time = np.zeros( data.shape[1])
        #for i in range( data.shape[1] ):
        #    if cell_names_all[i][-3] == '_':
        #        time[i] = int(cell_names_all[i][-2::] )
        #    else:
        #        time[i] = int(cell_names_all[i][-3::] )
        t = pd.read_csv(FileTime)

        if branch_no < 0:
            time = np.array( t.min(axis=1,skipna=True,numeric_only=True) )
            
        else:
            t = t.iloc[:,branch_no+1]
            t = np.array( t )
            time = t[ np.isnan(t) == False ] 
        
        
        tQuantiles = pd.qcut( time, q = Nt, duplicates ='drop')
        mid = [(a.left + a.right)/2 for a in tQuantiles]
        time = np.array( mid )

        time_pts = set( list(time) )
        time_pts = np.sort( list(time_pts) )
        dt = time_pts[1:] - time_pts[0:len(time_pts)-1]
        
     
        
        
        counts = data#np.log( data + 1)
        
        
        n = counts.shape[0]
        
        counts_all = [[0]]*Nt
        
        
        
        labels = np.zeros((1,counts.shape[1]))
        
        
        nums = [0]*Nt
        for i in range(Nt):
            idx = np.where(time == time_pts[i])[0]
            counts_all[i] = torch.tensor( counts[:,idx])
            nums[i] = len(idx)
        print(nums)
        s_cum = np.array( [0]+list( np.cumsum( nums)) )
        for i in range(Nt):
            labels[0,s_cum[i]:s_cum[i+1]] = i
        group_labels = list(range(Nt))
        for i in range(Nt):
            group_labels[i] = str(group_labels[i])
        counts = counts_all[0]
        for j in range(Nt-1):
            counts = np.concatenate( (counts, counts_all[j+1] ), axis=1 )

        # Find velocity

        n = counts.shape[0]
        counts_pca, pca = visualize_pca(counts,labels,group_labels,viz_opt='pca')

        
        
        Ts_prior,_ = solve_prior(counts,counts, Nt, labels, eps_samp=eps_samp, alpha=alpha)
        
        
        for i in range(Nt-1):
            Ts_prior[i] = Ts_prior[i]/Ts_prior[i].sum()
        
        velocities_all, velocities_all_signed = solve_velocities( counts_all, Ts_prior, order = 1, dt=dt)
        velocities = np.zeros( counts.shape )
        velocities_signed = np.zeros( counts.shape )
        
        for i in range(Nt):
            
            velocities[:,s_cum[i]:s_cum[i+1]] = velocities_all[i]
            velocities_signed[:,s_cum[i]:s_cum[i+1]] = velocities_all_signed[i]

        for l1 in l1_ratios:
            for a in lambdas:
                import copy
                velocities_all_signed_copy = copy.deepcopy( velocities_all_signed )
                velocities_signed_copy = copy.deepcopy( velocities_signed )
                
                Tv_total = OT_lagged_correlation(velocities_all_signed_copy, velocities_signed_copy, Ts_prior, stimulation=False, elastic_Net=True,tune=False,l1_opt=l1, alpha_opt=a, signed=True )
                if branch_no == -1:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/Ours_'+str(s)+'_lam_'+str(a)+'_l1_'+str(l1)+'.npy' 
                else:
                    FileOurs = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/Ours_'+str(s)+'_lam_'+str(a)+'_l1_'+str(l1)+'_'+str(branch_no)+'.npy' 
                np.save(FileOurs, Tv_total)



def load_others_result_BEELINE(example, method, seeds, eps_samp=None, alpha=None, eps_feat=None, dropout=None, combine='max',l1=None,lam=None, sign = None, penalty='EN',branch=True ):
    print(str(dropout)+' '+',branch '+str(branch) )
    if method == 'OTVelo-Corr' or method == 'OTVelo-Granger':
        eps_feat = None
        combine = 'sum'
    if method == 'GENIE3':
        branch = False
        
    if method == 'SINCERITIES' or method == 'HARISSA' or method =='CARDAMOM':
        branch = True
    AUROC = [0]*len(seeds)
    AUPRC = [0]*len(seeds)
    EP = [0]*len(seeds)
    counter = 0
    
    FileName = '../Data/Curated/'+example+'/'+example+'-2000-'+str(1)+'/ExpressionData.csv'
        
    data = pd.read_csv(FileName)
    cell_names_all = list( data.columns)
    cell_names_all = cell_names_all[1::]
    gene_names = list(data.iloc[:,0])
    n = len(gene_names)
    #print(len(gene_names))
    
    for s in seeds:
        
        if dropout == None:
            FileTruth = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/refNetwork.csv'
            FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/PseudoTime.csv'
        else:
            FileTruth = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/refNetwork.csv'
            FileTime = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/PseudoTime.csv'
        
        Tv_true = construct_truth_BEELINE( FileTruth, gene_names, sign = None)
        
        if dropout == None:
            
            FileResult = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/'+method
        else:
            FileResult = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'-'+str(dropout)+'/'+method
        
        if branch == False:
            
            if method == 'OTVelo-Corr' or method == 'OTVelo-Granger':
                if alpha != None and eps_samp != None and eps_feat == None:
                    FileResult = FileResult + '_alpha_'+str(alpha)+'_epscell_'+str(eps_samp)
                elif alpha != None and eps_samp != None:
                    FileResult = FileResult + '_alpha_'+str(alpha)+'_epscell_'+str(eps_samp)+'_epsgene_'+str(eps_feat)
                if method == 'OTVelo-Granger':
                    FileResult = FileResult + '_penalty'+penalty
                FileResult = FileResult+'.npy'
            else:
                
                if method != 'GENIE3':
                    FileResult = FileResult+'_score.npy'
                else:
                    FileResult = FileResult+'_score.txt'
                    
            if method == 'Ours' and lam != None:
                FileResult = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/Ours_'+str(s)+'_lam_'+str(lam)+'_l1_'+str(l1)+'.npy' 
            if method != 'GENIE3':
                Tv_total = np.load(FileResult)
            else:
                Tv_total = np.loadtxt(FileResult,delimiter=",")
            if Tv_total.shape[0] > n:
                Tv_total = Tv_total[1:,1:]
        else:
            if example == 'mCAD' or example == 'GSD':
                branches = 2
            elif example == 'HSC':
                branches = 4
            else:
                branches = 5
            Tv_total_branch = [ [0] ]*branches
            FileResult_branch = FileResult
            for br in range(branches):
                Tv_total_branch[br] = np.zeros( (n,n) )
                
                if alpha != None and eps_samp != None and eps_feat == None and br == 0:
                    FileResult = FileResult + '_alpha_'+str(alpha)+'_epscell_'+str(eps_samp)
                elif alpha != None and eps_samp != None and br == 0:
                    FileResult = FileResult + '_alpha_'+str(alpha)+'_epscell_'+str(eps_samp)+'_epsgene_'+str(eps_feat)
                #print(FileResult)
                FileResult_branch = FileResult
                
                if method == 'OTVelo-Corr' and lam != None:
                    FileResult_branch = '../Data/Curated/'+example+'/'+example+'-2000-'+str(s)+'/Ours_'+str(s)+'_lam_'+str(lam)+'_l1_'+str(l1)
                if method == 'OTVelo-Granger':
                    
                    FileResult_branch = FileResult_branch + '_penalty'+penalty
                    
                if not method in ['SINCERITIES','HARISSA','CARDAMOM'] :
                    
                    FileResult_branch = FileResult_branch+'_'+str(br)+'.npy'
                    
                elif method != 'SINCERITIES':
                    FileResult_branch = FileResult_branch+'_score_'+str(br)+'.npy'
                else:
                    FileResult_branch = FileResult_branch+'_score_'+str(br)+'.txt'
                #print(FileResult_branch)    
                #print(FileResult)
                            
                if method == 'SINCERITIES':
                    Tv_total_branch[br] = np.loadtxt(FileResult_branch,delimiter=",")
                else:
                    Tv_total_branch[br] = np.load(FileResult_branch)
                Tv_total_branch[br] = Tv_total_branch[br] - np.diag( np.diag(Tv_total_branch[br]))
                if Tv_total_branch[br].shape[0] > n:
                    #print(if Tv_total_branch[br].shape)
                    #print(method)
                    Tv_total_branch[br] = Tv_total_branch[br][1:,1:]
                #Tv_total_branch[br] = Tv_total_branch[br]/abs(Tv_total_branch[br]).max()
                #print(FileResult_branch)
                #print(FileResult_branch)
            Tv_total = np.zeros( (n,n))
            for i in range(n):
                for j in range(n):
                    if j != i:
                        for br in range(branches):
                            if combine == 'max':
                                if abs(Tv_total[i,j]) < abs(Tv_total_branch[br][i,j]):
                                    
                                    Tv_total[i,j] = Tv_total_branch[br][i,j]#/abs(Tv_total_branch[br][i,j]).max()
                                    
                            else:
                                Tv_total[i,j] = Tv_total[i,j] + Tv_total_branch[br][i,j]
            
        
        # if method == 'Ours_corr':
        #     plt.subplot(1,2,1)
        #     plt.imshow(Tv_total)
        #     Tv_total = np.zeros( (n,n) )
        #     for br in range(branches):
        #         Tv_total = Tv_total + Tv_total_branch[br]
        #     plt.subplot(1,2,2)
        #     plt.imshow(Tv_total)
        
        
        # print(Tv_total.shape)
        #print(sign)
        AUPRC[counter], AUROC[counter], random =  evaluate_AUC( Tv_total, Tv_true, sign = sign )
        
        
        EP[counter] = early_precision( Tv_total, Tv_true, sign=sign)
        if sign == None or sign == 'signed':
            random = np.mean( abs(Tv_true ) )
        elif sign == 'pos':
            random = np.mean( Tv_true>0  )
        else:
            random = np.mean( Tv_true<0 )
        # print('random AUPR='+str(random ) )

        counter += 1
        # print( Tv_true)
        # print(Tv_total)
   
    return AUROC, AUPRC, EP, random



