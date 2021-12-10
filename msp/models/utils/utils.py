from networkx.algorithms.shortest_paths.weighted import negative_edge_cycle
from networkx.utils import decorators
from networkx.utils.misc import flatten
import numpy as np
import torch
import torch.nn
import math
import torch.nn.functional as F

def get_non_zero(id_with_zero):
    mask = (id_with_zero != 0)
    id_without_zero = id_with_zero[mask]

    return id_without_zero

def generate_label_matrix(accident):
    label_matrix = torch.zeros((accident.shape[0], 40), device=torch.cuda.current_device())
    pos = torch.tensor([-math.exp(-max(0, (36-t) / 25)) for t in range(40)], device=torch.cuda.current_device())
    neg = torch.tensor([-1. for t in range(40)], device=torch.cuda.current_device())
    label_matrix[accident==True] = pos
    label_matrix[accident==False] = neg
    
    return label_matrix 

def caculate_loss(pred, labels):

    weight_loss = torch.mul(torch.log(pred), labels).sum(dim=1).mean()

    return weight_loss

def caculate_softmax_e_loss(pred, accident):
    
    crterion = torch.nn.CrossEntropyLoss(reduction='none')
    #positive sample
    frames = pred.shape[1]
    pos = pred[accident==True]
    pos_target = torch.tensor([1. for x in range(frames)], dtype=torch.long, device=torch.cuda.current_device())
    pos_penalty = torch.tensor([math.exp(-max(0, (76-t)/ 25)) for t in range(frames)], device=torch.cuda.current_device(), dtype=torch.long)
    all_positive_loss = []
    for batch in range(pos.shape[0]):
        loss = -crterion(pos[batch], pos_target)
        positive_loss = -torch.mul(loss, pos_penalty).sum()
        all_positive_loss.append(positive_loss)
    if len(all_positive_loss)!=0:
        all_positive_loss_t = torch.stack(all_positive_loss).mean()
    else:
        all_positive_loss_t = torch.tensor(0., dtype=torch.float32).type_as(pred)

    #neg
    neg = pred[accident==False]
    neg_target = torch.tensor([0. for x in range(frames)], dtype=torch.long, device=torch.cuda.current_device())
    all_neg_loss = []

    for batch in range(neg.shape[0]):
        loss_neg = crterion(neg[batch], neg_target).sum()
        all_neg_loss.append(loss_neg)
    if len(all_neg_loss) != 0:
        all_neg_loss_t = torch.stack(all_neg_loss).mean()
    else:
        all_neg_loss_t = torch.tensor(0., dtype=torch.float32).type_as(pred)

    pred_loss = all_positive_loss_t + 0.2 * all_neg_loss_t
    return pred_loss

def caculate_masker_gate_loss(masker_pred):

    pred_results = masker_pred[:, :2]
    label = masker_pred[:, -1]

    crterion = torch.nn.CrossEntropyLoss(reduction='none')
    go_away_pred = pred_results
    go_away_label = label
    go_away_loss = crterion(go_away_pred, 1 - go_away_label.long()).sum()

    shelter_pred = pred_results
    shelter_label = label
    shelter_loss = crterion(shelter_pred, shelter_label.long()).sum()
    
    loss_masker = 0.02 * go_away_loss + 0.02 * shelter_loss
  
    return loss_masker


def select_gate_feature(feature_bank, missing_status):

    go_away_index = torch.where(missing_status==1.0)
    go_away_feature = feature_bank[go_away_index]
    go_away_label = torch.zeros((go_away_index[0].shape[0], )).type_as(feature_bank)
    # 处理遮挡状态的物体，给1的标签
    shelter_index = torch.where(missing_status==2.0)
    shelter_feature = feature_bank[shelter_index]
    shelter_label = torch.ones((shelter_index[0].shape[0], )).type_as(feature_bank)
    # 合并特征和标签
    missing_feature = torch.cat((go_away_feature, shelter_feature), dim=0)
    labels = torch.cat((go_away_label, shelter_label), dim=0)

    return missing_feature, labels