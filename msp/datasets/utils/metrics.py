import numpy as np


def evaluate_metric(pred_np, label, fps=25):
    all_pred = pred_np[:,:,1] # shape [batch_size, frames]
    accident_status = label['accident'].cpu().numpy() # shape [batch_size, ]
    accident_time = 76
    pred_eval = []
    min_pred = np.inf
    n_frames = 0


    # access the frames before accident
    for idx, toa in enumerate(accident_status):
        if toa == True:
            pred = all_pred[idx, :int(accident_time)] # positive video
        else:
            pred = all_pred[idx, :] # negtive video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        pred_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps
    # iterate a set of thresholds from the minimum predications
    threholds = np.arange(max(min_pred,0), 1.0, 0.001)
    threholds_num = threholds.shape[0]
    Precision = np.zeros((threholds_num))
    Recall = np.zeros((threholds_num))
    Time = np.zeros((threholds_num)) 
    cnt = 0
    for Th in threholds:
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0 # number of TP videos
        # iterate each video sample
        for i in range(len(pred_eval)):
            # ture positive frames: (pred->1) & (gt->True)
            tp = np.where(pred_eval[i] * accident_status[i] >=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(accident_time)
                counter = counter + 1
            Tp_Fp += float(len(np.where(pred_eval[i]>=Th)[0])>0)

        if Tp_Fp == 0:
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(accident_status)==0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(accident_status)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]

    p_r_plot = {'precision':Precision,
                'recall':Recall}

    Time = Time[new_index]
    # Unique the recall 
    _, rep_index = np.unique(Recall, return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    if rep_index.size != 0:
        new_Time[-1] = Time[rep_index[-1]]
        new_Precision[-1] = Precision[rep_index[-1]]
        new_Recall = Recall[rep_index]
    else:
        new_Recall = Recall
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    #print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    #print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))
    
    return AP, mTTA, TTA_R80, p_r_plot


