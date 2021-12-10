import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch, batch
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import itertools
import math
import networkx


from ..builder import SGMODEL


@SGMODEL.register_module()
class SpatialGraphBatch(nn.Module):

    def __init__(self,
                 node_feature: int,
                 hidden_feature: int,
                 out_feature: int,
                 ):

        super(SpatialGraphBatch, self).__init__()
        
        self.g_conv1 = GCNConv(node_feature, hidden_feature)
        self.g_conv2 = GCNConv(hidden_feature, out_feature)

    def forward(self, 
                param_dict,
                module):

        if module == "Train":
            enc = self.gate_training(param_dict["feature_all"],
                            param_dict["graph_index"],
                            param_dict["graph_weight"])

        elif module == "Test":
            enc = self.module_testing(param_dict["location_t"],
                            param_dict["feature_t"],
                            param_dict["mask_t"],
                            param_dict["past_location_t"])
        return enc
        
    def gate_training(self,
                feature_all,
                graph_index,   
                graph_weight): 


        feature_flatten = torch.flatten(feature_all, start_dim=0, end_dim=1)
        graph_weight_flatten = torch.flatten(graph_weight, start_dim=0, end_dim=1)
        single_graph = graph_index[0]
        train_loader = self.generate_gcn_batch(feature_flatten, single_graph, graph_weight_flatten)
        for batch in train_loader:
            all_spatial_feature = F.sigmoid(self.g_conv1(batch.x, batch.edge_index, batch.edge_attr))
            env_feature = F.sigmoid(self.g_conv2(all_spatial_feature, batch.edge_index, batch.edge_attr))
            env_feature_re = env_feature.reshape(feature_all.shape[0], feature_all.shape[1], 19, 512)
        
        return env_feature_re
        

    def generate_gcn_batch(self, 
                           x,     
                           edge_index,  
                           edge_weight): 
        data_list = []

        for i in range(x.shape[0]):

            temp_data = Data(x=x[i], edge_index=edge_index, edge_attr=edge_weight[i])
            data_list.append(temp_data)

        loader = DataLoader(data_list, batch_size=x.shape[0], shuffle=False)
        
        return loader

    def module_testing(self, 
                location_t,
                feature_t,
                mask,
                past_location_t):

        num_boxes = location_t.shape[1]
        graph_edges = self.generate_graph_from_list(range(num_boxes))
        graph_weight = self.generate_weight(location_t, graph_edges, mask, past_location_t)
        
        graph_edges = torch.tensor(graph_edges).type_as(location_t).t().contiguous().type(torch.long)
        graph_edges_b = torch.stack([graph_edges for x in range(location_t.shape[0])], dim=0)
        graph_weight = graph_weight.type_as(location_t)

        out_batch = []
        for i in range(graph_edges_b.size(0)):
            hidden_enc_feature = F.sigmoid(self.g_conv1(feature_t[i], graph_edges_b[i], graph_weight[i]))
            enc_feature = F.sigmoid(self.g_conv2(hidden_enc_feature, graph_edges_b[i], graph_weight[i]))
            out_batch.append(enc_feature)

        out_batch = torch.stack(out_batch)

        return out_batch 
        
    
    def generate_weight(self, location_t, graph_edges, mask, past_location_t):

        weights = torch.zeros((location_t.shape[0], len(graph_edges), ), dtype=torch.float32).type_as(location_t)

        for i, edge in enumerate(graph_edges):

            c1 = [0.5 * (location_t[:, edge[0], 1] + location_t[:, edge[0], 3]),
                  0.5 * (location_t[:, edge[0], 2] + location_t[:, edge[0], 4])]
            c2 = [0.5 * (location_t[:, edge[1], 1] + location_t[:, edge[1], 3]),
                  0.5 * (location_t[:, edge[1], 2] + location_t[:, edge[1], 4])]
            d = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            # md
            md = self.get_motion_distance(edge[0], edge[1], past_location_t)
            weights[:, i] = torch.exp(-(0.7*d + 0.3*md))
            # mask empty position
            batch_index = ((mask[:,edge[0]]==True) & (mask[:,edge[1]]==True))
            weights[batch_index==False, i] = 0

        weights_nor = (weights - torch.min(weights, dim=1, keepdim=True)[0]) / (torch.max(weights, dim=1, keepdim=True)[0] - torch.min(weights, dim=1, keepdim=True)[0])
        
        return weights_nor

    def get_motion_distance(self, id_0, id_1, past_location):

        center_point = past_location[:,:,:,3:-1] - past_location[:,:,:,1:3]
        center_point_n_x = (center_point[:,:,:,0] * 1280).unsqueeze(dim=-1)
        center_point_n_y = (center_point[:,:,:,1] * 720).unsqueeze(dim=-1)
        center_point_n = torch.cat((center_point_n_x, center_point_n_y), dim=-1)

        delta_0 = (center_point_n[:, -1, id_0, 0] - center_point_n[:, 0, id_0, 0], 
                   center_point_n[:, -1, id_0, 1] - center_point_n[:, 0, id_0, 1])
        delta_1 = (center_point_n[:, -1, id_1, 0] - center_point_n[:, 0, id_1, 0],
                   center_point_n[:, -1, id_1, 1] - center_point_n[:, 0, id_1, 1])
        
        length_0 = (delta_0[0]**2 + delta_0[1]**2)**0.5
        length_1 = (delta_1[0]**2 + delta_1[1]**2)**0.5

        with torch.no_grad():
            length_0[length_0==0.] = 1.0
            length_1[length_1==0.] = 1.0

        nor_delta_x_0 = delta_0[0] / length_0
        nor_delta_y_0 = delta_0[1] / length_0

        nor_delta_x_1 = delta_1[0] / length_1
        nor_delta_y_1 = delta_1[1] / length_1
  
        d1 = ((center_point_n[:, 0, id_0, 0] + nor_delta_x_0 - center_point_n[:, 0, id_1, 0] - nor_delta_x_1)**2 +  \
              (center_point_n[:, 0, id_0, 1] + nor_delta_y_0 - center_point_n[:, 0, id_1, 1] - nor_delta_y_1)**2)**0.5

        d2 = ((center_point_n[:, 0, id_0, 0] - center_point_n[:, 0, id_1, 0])**2 + \
              (center_point_n[:, 0, id_0, 1] - center_point_n[:, 0, id_1, 1])**2)**0.5

        # motion distance
        md = d1 - d2

        return md


    def generate_graph_from_list(self, L, create_using=None):
        G = networkx.empty_graph(len(L), create_using)
        if len(L) > 1:
            if G.is_directed():
                edges = itertools.permutations(L,2)
            else:
                edges = itertools.combinations(L,2)

            G.add_edges_from(edges)

        graph_edges = list(G.edges())

        return graph_edges
