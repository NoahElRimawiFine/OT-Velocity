from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def signed_AUPRC( Tv_total, Tv_true ):
    if len(Tv_total.shape) == 2 and Tv_total.shape[0] == Tv_total.shape[1]:
        Tv_total = Tv_total - np.diag( np.diag(Tv_total) )
        Tv_true = Tv_true - np.diag( np.diag(Tv_true) )
    if abs(Tv_total).max() > 0:
        Tv_total = Tv_total/abs(Tv_total).max()
    
    n_thr = 1001
    thr = np.linspace(0,1,n_thr, endpoint=True)
    
    tp = np.zeros( (1,n_thr))
    fp = np.zeros( (1,n_thr))
    fn = np.zeros( (1,n_thr))
    tn = np.zeros( (1,n_thr))
    fpr = np.zeros( (n_thr))
    precision = np.zeros( (n_thr))
    recall = np.zeros( (n_thr))
    for i in range(n_thr):
        sign_info = np.sign( Tv_total )
        sign_info[ sign_info == 0 ] = 1
        adj_m = ( abs(Tv_total)>=thr[i] ) * sign_info
        compare = abs( adj_m + 3*Tv_true)
        #print(compare.shape)
        fp[0,i] = len( np.where( (compare<3)*(compare > 0)==1)[0] )
        #print( np.where( (compare<3)*(compare > 0)==1)[0] )
        tp[0,i] = len( np.where( compare==4)[0] )
        fn[0,i] = len( np.where( compare==3)[0] )
        tn[0,i] = len( np.where( compare==0)[0] )
        if tp[0,i] + fp[0,i] > 0 and tp[0,i] + fn[0,i]>0 and fp[0,i]+tn[0,i]>0:
            precision[i] = tp[0,i]/( tp[0,i] + fp[0,i] )
            recall[i] = tp[0,i]/( tp[0,i] + fn[0,i] )
            fpr[i] = fp[0,i]/( fp[0,i] + tn[0,i] )
    recall = recall[::-1]
    precision = precision[::-1]
    fpr = fpr[::-1]
    recall = np.concatenate( ([0.], recall) )
    precision = np.concatenate( ([1.], precision) )
    fpr = np.concatenate( ([0.], fpr) )
    idx = np.argsort( recall )
    recall_new = recall[idx]
    precision = precision[idx]
    
    # print(precision)
    AUPRC = np.trapz( precision, recall_new)

    idx = np.argsort( fpr )
    fpr = fpr[idx]
    recall = recall[idx]
    
    
    AUROC = np.trapz( recall, fpr)
    # plt.plot(precision,recall)
    # plt.show()
    # print(AUPRC)
    return AUPRC, AUROC#, precision, recall


def evaluate_AUC( Tv_total, Tv_true, sign = None ):
    #Tv_total = abs(Tv_total)
    #Tv_true = abs(Tv_true)
    #Tv_total = Tv_total / abs(Tv_total).max() 
    
    if Tv_true.shape[0] == Tv_true.shape[1]:
        n = Tv_total.shape[0]
    
        Tv_total_flattened = []
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_total_flattened += [Tv_total[i,j]]
        Tv_total_flattened = np.array( Tv_total_flattened )
    
        Tv_true_flattened = []
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_true_flattened += [Tv_true[i,j]]
        Tv_true_flattened = np.array( Tv_true_flattened )
    else:
        Tv_total_flattened = Tv_total
        Tv_true_flattened = Tv_true
    
    if sign == 'pos':
        Tv_total_flattened[Tv_total_flattened<0] = 0
        Tv_true_flattened[Tv_true_flattened<0] = 0

        if Tv_true_flattened.max() == 0 or Tv_total_flattened.max() == 0:
            AUPRC = 0
                
            AUROC = 0
        else:
            precision, recall, _ = precision_recall_curve( Tv_true_flattened,Tv_total_flattened)
            AUPRC = auc( recall, precision) 
                # print('AUPRC='+str( auc( recall, precision) )  )
                #Tv_small_total[0] = Tv_small_total[0]/Tv_small_total[0].max()
            AUROC = roc_auc_score( Tv_true_flattened,Tv_total_flattened)
    elif sign == 'neg':
        Tv_total_flattened[Tv_total_flattened>0] = 0
        Tv_total_flattened = np.abs(Tv_total_flattened)

        Tv_true_flattened[Tv_true_flattened>0] = 0
        Tv_true_flattened = np.abs(Tv_true_flattened)
        
        if Tv_true_flattened.max() == 0 or Tv_total_flattened.max() == 0:
            AUPRC = 0
                
            AUROC = 0
        else:
            precision, recall, _ = precision_recall_curve( Tv_true_flattened,Tv_total_flattened)
            AUPRC = auc( recall, precision) 
                # print('AUPRC='+str( auc( recall, precision) )  )
                #Tv_small_total[0] = Tv_small_total[0]/Tv_small_total[0].max()
            AUROC = roc_auc_score( Tv_true_flattened,Tv_total_flattened)
        
        
    elif sign == None:
        Tv_total_flattened = abs(Tv_total_flattened)
        Tv_true_flattened = abs(Tv_true_flattened)
    # else:
    #     for i in range( len(Tv_total_flattened) ):
    #         if Tv_total_flattened[i]*Tv_true_flattened[i] < 0:
    #             Tv_total_flattened[i] = 0
    #     Tv_total_flattened = abs(Tv_total_flattened)
    #     Tv_true_flattened = abs(Tv_true_flattened)      
        
        if Tv_true_flattened.max() == 0 or Tv_total_flattened.max() == 0:
            AUPRC = 0
                
            AUROC = 0
        else:
            precision, recall, _ = precision_recall_curve( Tv_true_flattened,Tv_total_flattened)
            AUPRC = auc( recall, precision) 
                # print('AUPRC='+str( auc( recall, precision) )  )
                #Tv_small_total[0] = Tv_small_total[0]/Tv_small_total[0].max()
            AUROC = roc_auc_score( Tv_true_flattened,Tv_total_flattened)
    else:
        if abs(Tv_true_flattened).max() == 0 or abs(Tv_total_flattened).max() == 0:
            AUPRC = 0
                
            AUROC = 0
        else:
            #print( np.sum( Tv_true ) )
            #print(np.sum(Tv_total))
            AUPRC, AUROC = signed_AUPRC(Tv_total_flattened, Tv_true_flattened )
            
            Tv_total_flattened = abs(Tv_total_flattened)
            Tv_true_flattened = abs(Tv_true_flattened)

            
            
            #AUROC = roc_auc_score( Tv_true_flattened,Tv_total_flattened)
            
    
    random = np.mean( abs(Tv_true_flattened ) )
    return AUPRC, AUROC, random


def early_precision( Tv_total, Tv_true, sign = None ):
    if Tv_true.shape[0] == Tv_true.shape[1]:
        n = Tv_total.shape[0]
    
        Tv_total_flattened = []
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_total_flattened += [Tv_total[i,j]]
        Tv_total_flattened = np.array( Tv_total_flattened )
    
        Tv_true_flattened = []
            
        for i in range(n):
            for j in range(n):
                if i != j:
                    Tv_true_flattened += [Tv_true[i,j]]
        Tv_true_flattened = np.array( Tv_true_flattened )
    else:
        Tv_total_flattened = Tv_total
        Tv_true_flattened = Tv_true

    if sign == None:
        k = abs( Tv_true_flattened ).sum()
    elif sign == 'pos':
        k = (Tv_true_flattened >0 ).sum()
    else:
        k = (Tv_true_flattened <0 ).sum()
        
    

    k_pred = ( abs(Tv_total_flattened)>0 ).sum()
    if k_pred < k:
        k = k_pred

    k = int(k)
    n_total = len(Tv_total_flattened)
    order = np.argsort( Tv_total_flattened )
    n_valid = 0
    # print( Tv_total_flattened)
    # print( Tv_true_flattened)
    for i in range(k):
        if Tv_true_flattened[ order[n_total - i -1] ] != 0:
            n_valid += 1

    if k == 0:
        return 0
    else:
        return n_valid/k

def evaluate_Jaccard(benchmark, algo, seeds=range(1,11)):
    inter = abs(np.load(f'../Data/HARISSA/{benchmark}/True/inter_1.npy'))
    G = inter.shape[0]

    # 1. Directed inference from snapshots
    edges = [(i,j) for i in range(G) for j in set(range(1,G))-{i}]
    y0 = np.array([inter[i,j] for (i,j) in edges])
    k = abs(y0).sum()
    
    y1 = [0]*len(seeds)
    for n,r in enumerate(seeds):
        y1[n] = [0]
        if algo == 'Ours':
            score = abs(np.load(f'../Data/HARISSA/{benchmark}/Ours/score_{r}penaltyl1.npy'))
        elif algo == 'Ours_l2':
            score = abs(np.load(f'../Data/HARISSA/{benchmark}/Ours/score_{r}penaltyl2.npy'))
        elif algo == 'Ours_LOOCV':
            score = abs(np.load(f'../Data/HARISSA/{benchmark}/Ours/score_{r}penaltytune.npy'))
        else:
            score = abs(np.load(f'../Data/HARISSA/{benchmark}/{algo}/score_{r}.npy'))
        if algo=='GENIE3': score = score.T # Solve direction problem?
        score = abs(score)  
        y1[n] = np.array([score[i,j] for (i,j) in edges])
        y1_sort = np.sort(y1[n])[::-1]
        if np.sum( y1_sort>1E-10) > k:
            threshold = y1_sort[int(k-1)]
        else:
            threshold = y1_sort[y1_sort>1E-10].min()
        y1[n][y1[n]<threshold] = 0
        y1[n][y1[n]>=threshold] = 1
        #print(y1[n].sum())
        #print(k)
    Jaccard_ind = []
    for n,r in enumerate(seeds):
        for n2, r2 in enumerate(seeds):
            if n2 > n:
                Jaccard_ind += [np.sum( y1[n]*y1[n2])/(  np.sum( y1[n]+y1[n2]>0.5  )) ]
    return Jaccard_ind  
  
