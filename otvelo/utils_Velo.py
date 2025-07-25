import pandas as pd
import numpy as np
#from coot_torch import *

import matplotlib.pyplot as plt
#from agw_scootr import *
#from agw_scootr_nogw import *
from utils import *
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

def compute_graph_distances(data, n_neighbors=5, mode="distance", metric="correlation"):
	"""
	
	"""
	graph=kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode, metric=metric, include_self=True)
	shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
	max_dist=np.nanmax(shortestPath[shortestPath != np.inf])
	shortestPath[shortestPath > max_dist] = max_dist

	return np.asarray(shortestPath)
def init_matrix_np(X1, X2, v1, v2):
	def f1(a):
		return (a ** 2)

	def f2(b):
		return (b ** 2)

	def h1(a):
		return a

	def h2(b):
		return 2 * b

	constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
					 np.ones(f1(X2).shape[0]).reshape(1, -1))
	constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
					 np.dot(v2.reshape(1, -1), f2(X2).T))

	constC = constC1 + constC2
	hX1 = h1(X1)
	hX2 = h2(X2)

	return constC, hX1, hX2

def init_matrix_GW(C1,C2,p,q,loss_fun='square_loss'):
	""" 
	"""        
	if loss_fun == 'square_loss':
		def f1(a):
			return a**2 

		def f2(b):
			return b**2

		def h1(a):
			return a

		def h2(b):
			return 2*b

	constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
					 np.ones(len(q)).reshape(1, -1))
	constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
					 np.dot(q.reshape(1, -1), f2(C2).T))
	constC=constC1+constC2
	hC1 = h1(C1)
	hC2 = h2(C2)

	return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):
	""" 
	"""
	A=-np.dot(hC1, T).dot(hC2.T)
	tens = constC+A

	return tens

def dist(x1, x2=None, metric='sqeuclidean'):
	"""
	Compute distances between pairs of samples across x1 and x2 using scipy.spatial.distance.cdist
	If x2=None, x2=x1, then we compute intra-domain sample-sample distances

	Parameters
	----------
	x1 : np.array (n1,d)-- A matrix with n1 samples of size d
	x2 : np.array (n2,d)-- optional. Matrix with n2 samples of size d (if None, then x2=x1)
	metric : str or function -- distance metric, optional
		If a string, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
		'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
		 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'. Full list in the doc of scipy
	Returns
	-------
	M : np.array (n1,n2) -- Distance matrix computed with the given metric
	"""
	if x2 is None:
		x2 = x1

	return cdist(x1, x2, metric=metric)

def vis_mean_level(counts,vis_legend = False,gene_of_interest = None, gene_names = None ):
    plt.rcParams["figure.figsize"] = (10,5)
    Nt = len(counts)
    n = counts[0].shape[0]
    if gene_of_interest is None:
        gene_of_interest = range(n)
    mean_level = np.zeros((Nt,n))

    sd_level = np.zeros((Nt,n))

    for i in range(Nt):

        for j in gene_of_interest:
            mean_level[i,j] = np.mean(counts[i][j,:])#.detach().numpy())
            sd_level[i,j] = np.std(counts[i][j,:])#.detach().numpy())


    plt.subplot(1,2,1)
    for i in gene_of_interest:
        plt.title('Mean expression level vs time')
        plt.plot(mean_level[:,i],'o-',label=str(i))
    #     plt.plot(mean_level[:,i]+sd_level[:,i],color='gray')
    #     plt.plot(mean_level[:,i]-sd_level[:,i],color='gray')
    plt.subplot(1,2,2)
    for i in gene_of_interest:
        if gene_names is None:
            plt.plot((mean_level[:,i] - np.min(mean_level[:,i])) /(np.max(mean_level[:,i])- np.min(mean_level[:,i])),'o-',label=str(i))
        else:
            plt.plot((mean_level[:,i] - np.min(mean_level[:,i])) /(np.max(mean_level[:,i])- np.min(mean_level[:,i])),'o-',label=gene_names[i])
        plt.title('Normalized to [0,1]')
    if n < 10 or vis_legend:
        plt.legend()
    plt.tight_layout()
    plt.show()
    return mean_level
    
def visualize_pca(counts, labels, group_labels, viz_opt='pca', viz=False, plot_separate=False):
    colors = ["r", "b", "g","c","y" ,"k"]
    markers = ['o','+','x','^','s','>']
    # plt.rcParams["figure.figsize"] = (5,5)
    # PCAs:
    if viz_opt == 'pca':
        from sklearn.decomposition import PCA
        plt.rcParams["font.size"] = 24
        reducer = PCA()
    elif viz_opt == 'umap':
        import umap.umap_ as umap
        reducer = umap.UMAP()
    Nt = len(group_labels)
    # counts.T/np.reshape(counts.T.sum(axis=1),(224,1))
    # Xt = pca.fit_transform(counts.T/np.reshape(counts.T.sum(axis=1),(224,1)) )
    # Xt = pca.fit_transform(torch.tensor(normalize(counts.T)))
    Xt = reducer.fit_transform(counts.T )
    if viz:
        for i in range(Nt):
            idx = np.where(labels.ravel() == i)
            if plot_separate:
                plt.subplot(1,Nt,i+1)
                if viz_opt == 'pca':
                    plt.xlabel('PC1')
                else:
                    plt.xlabel('UMAP1')
                if i == 0:
                    if viz_opt == 'pca':
                        plt.ylabel('PC2')
                    else:
                        plt.ylabel('UMAP2')
                plt.title('Data at t='+str(i))
                
            plt.scatter(Xt[idx,0],Xt[idx,1],label=group_labels[i])
            plt.xlim( [ Xt[:,0].min(), Xt[:,0].max()] )
            plt.ylim( [ Xt[:,1].min(), Xt[:,1].max()] )
            
        # plt.colorbar()
        # plt.xlim([-0.3,0.4])
        # plt.ylim([-0.3,0.3])
       
        if plot_separate == False:
            if viz_opt == 'pca':
                plt.xlabel('PC1')
                plt.ylabel('PC2')
            else:
                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')
            plt.title('Data')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return Xt, reducer
    # plt.savefig('figures/scGEM/counts_PCA.png', bbox_inches='tight')
#    plt.show(block=False)
    

   


    
def visualize_cell_path(counts, counts_all, group_labels, Ts, start_time=0, marker_size=20, viz_opt='pca', viz_bytime = False):
    nums = []
    Nt = len(counts_all)
    for i in range(Nt):
        nums = nums + [ counts_all[i].shape[1] ]
    nums_total = np.cumsum(nums)
    
    
    Ts_cum = [Ts[0]]*(Nt-1)
    for i in range(1,Nt-1):
    
        Ts_cum[i] = np.matmul( Ts_cum[i-1], Ts[i] )*nums[i]
        
    
    
    counts_forward = np.zeros( ( np.shape(counts_all[start_time])[0] ,Nt*np.shape(counts_all[start_time])[1] ) )


    for j in range(start_time,start_time+1):

        counts_forward[:,(j)*np.shape(counts_all[start_time])[1]:(j+1)*np.shape(counts_all[start_time])[1]] = counts_all[start_time]

    labels_forward = start_time + np.zeros( (1,Nt*np.shape(counts_all[start_time])[1]))

    start_size = np.shape(counts_all[start_time])[1]
    for i in range(start_time + 1,Nt ):

        counts_forward[:,start_size*i:start_size*(i+1) ] = nums[start_time]*np.matmul( counts_all[i],Ts_cum[i-1].T )
        labels_forward[0,start_size*i:start_size*(i+1)] = i



    viz_traj = 1

    plt.rcParams["figure.figsize"] = (5,5)
    # PCAs:
    from sklearn.decomposition import PCA
    plt.rcParams["font.size"] = 24
    if viz_opt == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA()

        pca.fit_transform(counts.T )
        New_Xt = pca.transform(counts_forward.T)
    else:
        import umap.umap_ as umap
        reducer = umap.UMAP()
        reducer.fit_transform(counts.T)
        New_Xt = reducer.transform(counts_forward.T)

    
    


    for i in range(Nt):
        idx = np.where(labels_forward == i)[1]

        plt.scatter(New_Xt[idx,0],New_Xt[idx,1],marker_size,label=group_labels[i])

        if viz_traj and i<Nt-1:
            for j in range(len(idx)):
                plt.plot( [New_Xt[idx[j],0], New_Xt[idx[j]+len(idx),0]],[New_Xt[idx[j],1], New_Xt[idx[j]+len(idx),1]],color='k',alpha=0.1)
            if viz_bytime:
                idx = np.where(labels_forward == i+1)[1]
                plt.scatter(New_Xt[idx,0],New_Xt[idx,1],marker_size,label=group_labels[i])
                plt.title('time '+str(i))
                plt.show()

    return counts_forward, New_Xt


def solve_prior(counts, counts_pca, Nt, labels, eps_samp, alpha=0.5):
    Ts = [[0]]*(Nt-1)
    log = [[0]]*(Nt-1)
    n = counts.shape[0]
    import ot
    from ot.gromov import entropic_fused_gromov_wasserstein
    
    for t in range(Nt-1):
            
        
        idx_t = np.where(labels == t)[1]
        X1 = counts[:,idx_t].T
        idx_t1 = np.where(labels == t+1)[1]
        X2 = counts[:,idx_t1].T

        # This is for global OT distance
        # from utils import compute_graph_distances
        M = ot.dist( counts_pca[:,idx_t].T, counts_pca[:,idx_t1].T)
        #M = ot.dist( X1, X2)
        #print(X1.shape)
        M = M/M.max()
        
        
        # This is the part for GW distance:
        n_neighbors = np.max( (1,np.min( ( int(0.2*X1.shape[0] ), int(0.2*X2.shape[0]), 50) ) ) )
        D1 = compute_graph_distances(X1, n_neighbors = n_neighbors, metric="euclidean")
        D2 = compute_graph_distances(X2, n_neighbors = n_neighbors, metric="euclidean")

        if D1.max() == 0 or D2.max() == 0:
            alpha = 0
        else:
            D1 = D1/D1.max()
            D2 = D2/D2.max()
        
        Ts[t], log_t = entropic_fused_gromov_wasserstein( M, D1, D2, epsilon = eps_samp, alpha=alpha, log=True)
        if np.isnan(Ts[t][0,0]) or  abs(Ts[t].sum()-1)>1E-3 :
            Ts[t] = np.ones( Ts[t].shape)
            Ts[t] = Ts[t]/Ts[t].sum()
        
        log[t] = log_t['fgw_dist']
    return Ts, log




def solve_velocities( counts_all, Ts_prior, order = 1, idx_marker = None, dt=None, stimulation=False):

    Nt = len(counts_all)
    if dt is None:
        dt = [1]*Nt
    
    n = counts_all[0].shape[0]
    if idx_marker == None:
        idx_marker = list(range(n))
    velocities_all = [ [0] ]*(Nt)
    velocities_all_signed = [ [0] ]*(Nt)
    nums = []
        
    for i in range(Nt):
        nums = nums + [ counts_all[i].shape[1] ]
   

    i = 0
    
    velocities_all[i] = [0]
    
    count_t_mapped = np.matmul( counts_all[i+1][idx_marker,:],Ts_prior[i].T/np.sum(Ts_prior[i].T,axis=0)  )
    
    
    velocities_all[i] = (count_t_mapped - counts_all[i][idx_marker,:] )/dt[0]

    if order > 1:
        coupling2 = Ts_prior[i].dot( Ts_prior[i+1] )
        count_mapped2 = np.matmul( counts_all[i+2][idx_marker,:],coupling2.T/np.sum(coupling2.T,axis=0)  )
        velocities2 = (count_mapped2[idx_marker,:] - counts_all[i][idx_marker,:])/(dt[0]+dt[1])
        velocities_all[i] = (dt[0]+dt[1])/(dt[1])*velocities_all[i] - dt[0]/dt[1]*velocities2
        
    if stimulation:    
        
        velocities_all[0] = np.zeros( (n,nums[0]))
        velocities_all[0][0,:] = 1
        velocities_all_signed[0] = velocities_all[0]
    
        
        
    velocities_all_signed[i] = velocities_all[i]
    velocities_all[i] = np.abs(velocities_all[i])
    
    for i in range(1,Nt-1):
        velocities_all[i] = [0]
        count_t_mapped = np.matmul( counts_all[i+1][idx_marker,:],Ts_prior[i].T/np.sum(Ts_prior[i].T,axis=0) )
        
        velocities_all[i] = (count_t_mapped - counts_all[i][idx_marker,:])/dt[i]
        count_t_mapped = np.matmul( counts_all[i-1][idx_marker,:],Ts_prior[i-1]/np.sum(Ts_prior[i-1],axis=0) )
        
        velocities_all[i] = velocities_all[i]*(dt[i-1]/(dt[i]+dt[i-1])) + (counts_all[i][idx_marker,:] - count_t_mapped)/dt[i-1]*(dt[i]/(dt[i]+dt[i-1]))
        if stimulation:
            velocities_all[i][0,:] = 0
        
            
        velocities_all_signed[i] = velocities_all[i]
        velocities_all[i] = np.abs(velocities_all[i])
        
    
    i = Nt-1
    velocities_all[i] = [0]
    count_t_mapped = np.matmul( counts_all[i-1][idx_marker,:], Ts_prior[i-1]/np.sum(Ts_prior[i-1],axis=0)  )

    velocities_all[i] = ( counts_all[i][idx_marker,:] - count_t_mapped )/dt[i-1]

    if order > 1:
        coupling2 = Ts_prior[i-1].dot( Ts_prior[i-2] )
        count_mapped2 = np.matmul( counts_all[i-2][idx_marker,:],coupling2/np.sum(coupling2,axis=0)  )
        velocities2 = (count_mapped2[idx_marker,:] - counts_all[i-1][idx_marker,:])/(dt[i-1]+dt[i-2])
        velocities_all[i] = (dt[i-1]+dt[i-2])/(dt[i-1])*velocities_all[i] - dt[i-2]/dt[i-1]*velocities2
        
    if stimulation:
        velocities_all[i][0,:] = 0
    velocities_all_signed[i] = velocities_all[i]
    velocities_all[i] = np.abs(velocities_all[i])

    
    return velocities_all, velocities_all_signed

    


def OT_lagged_correlation(velocities_all_signed, velocities_signed, Ts_prior, normalization=True, stimulation=False, elastic_Net=False, alpha_opt=1.0, l1_opt=0.5,tune=False, g_s = None, g_t=None, return_slice=False, signed=False, lags=False):
    
    if signed == False:
        positive = True
    else:
        positive = False

    
    
    Nt = len(velocities_all_signed)
    n = velocities_all_signed[0].shape[0]
    #print(g_s)
    if g_s == None:
        g_s = range(n)
    if g_t == None:
        if stimulation:
            g_t = range(1,n)
        else:
            g_t = range(n)
    
    
    import copy
    
   
    velocities_all_signed_norm = copy.deepcopy(velocities_all_signed)
    
    if normalization:
        for i in range(n):
            
            for j in range(Nt):
                if stimulation == True and i > 0 and j > 0:
                   velocities_all_signed_norm[j][i,:] = velocities_all_signed[j][i,:]/np.sqrt(np.var( velocities_signed[i,velocities_all_signed_norm[0].shape[1]:]) ) 
                
                if np.var( velocities_signed[i,:]) > 0:
                    velocities_all_signed_norm[j][i,:] = velocities_all_signed[j][i,:]/np.sqrt(np.var( velocities_signed[i,:]) ) 
                
                
    
    
    corr_OT = np.zeros( (n,n) )
    corr_slice = [0]*(Nt-1)

    if elastic_Net == False:
        if lags == False:
            corr_slice = correlation_varied_lags(velocities_all_signed_norm, Ts_prior, n, g_s, g_t,  lag=1)
            for t in range(Nt-1):
                corr_OT = corr_OT + corr_slice[t]
        else:
            n_lags = 3#Nt-2
            corr_lag = [0]*n_lags
            corr_lag_overall = [0]*n_lags
            for lag in range(1,n_lags+1):
                corr_lag[lag-1] = []
                corr_lag[lag-1] = correlation_varied_lags(velocities_all_signed_norm, Ts_prior, n, g_s, g_t, lag=lag)
                corr_lag_overall[lag-1] = np.zeros( (n,n) )
                for j in range(Nt-1):
                    corr_lag_overall[lag-1] = corr_lag_overall[lag-1] + corr_lag[lag-1][j]
                #print(corr_lag_overall[lag-1])
            for i in range(n):
                for j in range(n):
                    for lag in range(1,1+n_lags):
                        if abs(corr_OT[i,j] ) < abs(corr_lag_overall[lag-1][i,j]):
                            
                            corr_OT[i,j] = corr_lag_overall[lag-1][i,j]
                
    for t in range( Nt-1 ):
        #print(t)
        
        
        
        if elastic_Net == True:
            
            # Just the correlation approach
            
            #corr_slice[t] = np.zeros( (n,n) )
            #if lags == False:
            #    corr_slice[t][np.ix_(g_s,g_t)] = np.dot( velocities_all_signed_norm[t][g_s,:],Ts_prior[t] ).dot( velocities_all_signed_norm[t+1][g_t,:].T )/(Nt-1)
            
            corr_slice[t] = np.zeros( (n,n))
            # Granger causality via LASSO
            
            from sklearn.linear_model import ElasticNet, Ridge
            if tune == False:
                if l1_opt == 0:
                    model = Ridge(alpha=alpha_opt,fit_intercept=False, positive=positive)
                else:
                    model = ElasticNet(alpha=alpha_opt,fit_intercept=False, positive=positive, l1_ratio=l1_opt)
                
                if signed == False:
                    model.fit(  abs(np.dot( velocities_all_signed_norm[t][g_s,:],Ts_prior[t]/np.sum(Ts_prior[t],axis=0)).T ), abs(velocities_all_signed_norm[t+1][g_t,:].T ) )
                else:
                    
                    model.fit(  np.dot( velocities_all_signed_norm[t][g_s,:],Ts_prior[t]/np.sum(Ts_prior[t],axis=0)).T, velocities_all_signed_norm[t+1][g_t,:].T  )
                    
                corr_slice[t] = np.zeros( (n,n) )
                corr_slice[t][np.ix_(g_s,g_t)] = model.coef_.transpose()
            else:
                corr_slice[t] = np.zeros( (n,n) )
                corr_slice[t] = OT_lagged_correlation_CV_each( velocities_all_signed_norm[t+1],np.dot( velocities_all_signed_norm[t],Ts_prior[t]/np.sum(Ts_prior[t],axis=0)) , g_s, g_t, n, signed=signed )
            
              
            corr_OT = corr_OT + corr_slice[t]
            
    if return_slice == False:
        return corr_OT
    else:
        return corr_OT, corr_slice


def correlation_varied_lags(velocities_all_signed_norm, Ts_prior, n, g_s, g_t, lag=1):
   
    Nt = len(Ts_prior) + 1
    corr_OT = np.zeros( (n,n) )
    corr_slice = [0]*(Nt-1)
    Ts_prior_lag = [0]*(Nt-1)
    for t in range(Nt-1):
        corr_slice[t] = np.zeros( (n,n))
                
    for t in range( Nt-lag ):
        
        
        Ts_prior_lag[t] = Ts_prior[t]
        for tt in range(1,lag):
            Ts_prior_lag[t] = Ts_prior_lag[t].dot( Ts_prior[t+tt] )
        Ts_prior_lag[t] = Ts_prior_lag[t]/Ts_prior_lag[t].sum()
        
        corr_slice[t] = np.zeros( (n,n) )
        corr_slice[t][np.ix_(g_s,g_t)] = np.dot( velocities_all_signed_norm[t][g_s,:],Ts_prior_lag[t] ).dot( velocities_all_signed_norm[t+lag][g_t,:].T )/(Nt-lag)
    return corr_slice



def OT_lagged_correlation_CV_each( velocities_pred, velocities_prior, g_s, g_t, n, signed=False ):
    
   
    corr_slice = np.zeros( ( n,n) )
    #alphas = np.geomspace(0.1,10,11)
    for i in g_t:
        from sklearn.linear_model import LassoCV, ElasticNetCV
        # model = LassoCV( cv=5, positive=True, random_state = 0, fit_intercept=False)#, alphas=alphas )
        n_cv = 5
        if velocities_pred.shape[1] < 5:
            n_cv = 2
        if signed == False:
            model = ElasticNetCV( cv=n_cv, positive=True, random_state = 0, fit_intercept=False)
            model.fit(  abs(velocities_prior[g_s,:].T), abs(velocities_pred[i,:].T) )
        else:
            model = ElasticNetCV( cv=n_cv, l1_ratio=[0,0.5,1.0],alphas=[0.1,0.4,0.7,1.0,1.3,1.6],positive=False, fit_intercept=False)
            model.fit(  velocities_prior.T, velocities_pred[i,:].T )
        corr_slice[:,i] = model.coef_.transpose()
    
    return corr_slice
