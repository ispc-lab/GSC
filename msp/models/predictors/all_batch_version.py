import torch
import torch.nn as nn
from mmcv.utils.config import ConfigDict
from ..utils import *
from ..builder import PREDICTORS, build_spatial, build_temporal, build_attn, \
                                  build_accident_block, build_gate, build_loss, build_nearfuture


@PREDICTORS.register_module()
class STAttnsGraphSimAllBatch(nn.Module):

    def __init__(self,
                 spatial_graph: ConfigDict,
                 temporal_graph:ConfigDict,
                 gate: ConfigDict,
                 temporal_attn: ConfigDict,
                 near_future: ConfigDict,
                 accident_block: ConfigDict,
                 loss: ConfigDict):

        super(STAttnsGraphSimAllBatch, self).__init__()

        self.spatial_graph = build_spatial(spatial_graph)
        self.temporal_graph = build_temporal(temporal_graph)
        self.mask_gate = build_gate(gate)
        self.near_future = build_nearfuture(near_future)
        self.temporal_attn = build_attn(temporal_attn)
        self.accident_block = build_accident_block(accident_block)
        self.loss_func = build_loss(loss)
        # self.phi_x = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.LeakyReLU(0.1)
        # )

    def forward(self, data, module="Train"):
        location = data['location'] # batch x frames x 19 x 6
        maskers = data["maskers"]   # batch x frames x 19 
        missing_status = data["missing_status"]  
        feature = data['feature'][:,:,:,1:]   
        batch_size = location.shape[0]
        h0 = torch.randn(2, batch_size, self.accident_block.hidden_feature).type_as(location)
        c0 = torch.randn(2, batch_size, self.accident_block.hidden_feature).type_as(location)
        location_past_index = torch.stack([location[:, 0] for x in range(5)], dim=1)
        location_past = torch.cat([location_past_index, location], dim=1)  # batch_size x (frames+5) x 19 x 6


        if module == "Train":
           graph_index = data["graph_index"]
           graph_weight = data["graph_weight"]
           train_dict = {
               "feature_all":feature,
               "graph_index":graph_index,
               "graph_weight":graph_weight}
           spatial_feature_bank = self.spatial_graph(train_dict, module) 

           train_dict_t = {
               "spatial_feature": spatial_feature_bank
           }
           temporal_feature_bank = self.temporal_graph(train_dict_t, module)

           select_feature, label = select_gate_feature(temporal_feature_bank, missing_status)
           train_gate_dict = {
               "select_feature":select_feature}
           pred_status = self.mask_gate(train_gate_dict, module)
           pred_label = torch.hstack((pred_status, label.unsqueeze(-1)))
           # weighted feature
           attn_feature = self.temporal_attn(temporal_feature_bank, module)  
           # LSTM
           pred_accident =  self.accident_block(attn_feature, h0, c0).transpose(1,0)
           
           return pred_accident, pred_label
           
        elif module == "Test":
            spatial_feature_bank = []
            temporal_feature_bank = []
            temporal_feature_single_bank = []
            masks = []
            masker_dict = {
                'index':[torch.empty((0,)).type_as(location), torch.empty((0,)).type_as(location)],
                'pred_label':torch.empty((0, 3)).type_as(location),
                'timestamp': [] # add timestamps
            }

            for t in range(location.size(1)):# pass throught time

                location_t = location[:, t]                  
                past_location_t = location_past[:, t:(t+5)] 
                # missing detect and fill the position
                mask_t = maskers[:, t]
                masker_param_dict={
                    "all_mask": masks,
                    "present_mask": mask_t,
                    "feature_bank": temporal_feature_single_bank,
                    "time":t,
                    "location_all":location,
                    "masker_dict":masker_dict}

                mask_t_filled, masker_dict, fill_position_index = self.mask_gate(masker_param_dict, module)

                masks.append(mask_t_filled)
   
                location_t_filled, feature_t_filled = self.near_future(location,     
                                                                    feature,   
                                                                    t,          
                                                                    fill_position_index)

                location[:, t].data = location_t_filled
                feature[:, t].data = feature_t_filled

                # GCN encoder
                test_param_dict = {
                    "location_t":location_t_filled,
                    "feature_t":feature_t_filled,
                    "mask_t":mask_t_filled,
                    "past_location_t":past_location_t
                }

                enc = self.spatial_graph(module="Test", param_dict=test_param_dict) 

                # save the spatial_bank
                spatial_feature_bank.append(enc)
                # temporal graph
                test_dict_t = {
                    "spatial_feature":spatial_feature_bank,
                    "mask":mask_t_filled}
                temporal_feature_t = self.temporal_graph(test_dict_t, module)

                temporal_feature_single_bank.append(temporal_feature_t)
                attn_feature_t = self.temporal_attn(temporal_feature_t, module)

                temporal_feature_bank.append(attn_feature_t)
  
            temporal_feature_bank = torch.stack(temporal_feature_bank, dim=0)

            # LSTM

            pred = self.accident_block(temporal_feature_bank, h0, c0).transpose(1,0)

            return pred, masker_dict['pred_label']

    def loss(self, pred, masker_pred, data):

        accident = data["accident"]
        pred_loss = caculate_softmax_e_loss(pred, accident)
        mask_gate_loss = caculate_masker_gate_loss(masker_pred)

        return pred_loss, mask_gate_loss
    
    def predict(self, data, module):
        with torch.no_grad():
            pred, pred_gate = self(data, module)
            pred_np = pred.cpu().numpy()
            pred_gate_np = pred_gate.cpu().numpy()
        
        return pred_np

