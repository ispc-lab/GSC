from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import math


from ..builder import TGMODEL


@TGMODEL.register_module()
class TemporalGraphBatch(nn.Module):
    def __init__(self,
                 past_num: int,
                 input_feature: int, 
                 out_feature: int):

        super(TemporalGraphBatch, self).__init__()

        self.past_num = past_num
        self.out_feature = out_feature
        self.in_feature = input_feature

        edge_index, edge_weight = self.generate_graph(self.past_num)

        self.register_buffer("t_edge_index", edge_index)
        self.register_buffer("t_edge_weight", edge_weight)

        self.conv1 = GCNConv(input_feature, out_feature)

    def forward(self, 
                param_dict,
                module):
        if module == "Train":
            enc = self.ts_train(param_dict["spatial_feature"]) 
        elif module == "Test":
            enc = self.ts_test(param_dict["spatial_feature"],
                               param_dict["mask"])

        return enc

    def ts_train(self,
                spatial_feature_bank 
                ):
        batch_size = spatial_feature_bank.shape[0]
        filled = spatial_feature_bank[:,0,:,:]
        pred_filled = torch.stack([filled for i in range(self.past_num-1)], dim=1)
        s_bank_fill = torch.cat((spatial_feature_bank, pred_filled), dim=1)
        s_bank_per = s_bank_fill.permute(0,2,1,3)
        s_bank_re = torch.flatten(s_bank_per, start_dim=0, end_dim=1)
        througth_time = torch.empty(size=[batch_size * 19, 0, self.past_num, self.in_feature]).type_as(spatial_feature_bank)

        for time_step in range(self.past_num-1,100+self.past_num-1):
            singe_step = s_bank_re[:,time_step-self.past_num+1:time_step+1,:].unsqueeze(dim=1)
            througth_time = torch.cat((througth_time, singe_step), dim=1)

        t_t_re = torch.flatten(througth_time, start_dim=0, end_dim=1)
        enc = self.conv1(t_t_re, self.t_edge_index, self.t_edge_weight)
        shape_1 = enc.reshape(batch_size*19, 100, self.past_num, self.out_feature)[:,:,-1,:] 
        shape_2 = shape_1.reshape(batch_size, 19, 100, self.out_feature)
        final = shape_2.permute(0,2,1,3)

        return final

    def ts_test(self, 
                spatial_feature_bank, 
                mask):

        spatial_feature_bank_tensor = torch.stack(spatial_feature_bank, dim=0).permute(1,0,2,3)
        temporal_feature_t = torch.zeros((mask.shape[0], 19, 512,), dtype=torch.float32).type_as(spatial_feature_bank_tensor)

        if spatial_feature_bank_tensor.shape[1] < self.past_num:
            missing_num = self.past_num - spatial_feature_bank_tensor.shape[1]
            unpack = torch.stack([spatial_feature_bank_tensor[:,0,:,:] for x in range(missing_num)], dim=1)
            spatial_feature_bank_pack = torch.cat([unpack, spatial_feature_bank_tensor], dim=1)
        else:
            spatial_feature_bank_pack = spatial_feature_bank_tensor[:,-5:]

        sf_pack = spatial_feature_bank_pack.permute(0,2,1,3)

        sf_pack_f = torch.flatten(sf_pack, start_dim=0, end_dim=1)
        t_f_t_all = self.conv1(sf_pack_f, self.t_edge_index, self.t_edge_weight)
        t_f_t_all_re = t_f_t_all.reshape(mask.shape[0],19,5,512).permute(0,2,1,3)[:,-1,:,:]
        temporal_feature_t[torch.where(mask==True)] = t_f_t_all_re[torch.where(mask==True)]

        return temporal_feature_t

    def generate_graph(self, 
                       past_num: int):

        edge_index = []
        edge_weight = []

        for i in range(past_num - 1):
            edge_index.append([i, past_num - 1])
            edge_weight.append(math.exp(-(past_num - 1 -i)))
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_weight, dtype=torch.float32)