import torch
import torch.nn as nn
from ..builder import GATEMODEL


@GATEMODEL.register_module()
class MaskGate(torch.nn.Module):
    def __init__(self, past_num):
        
        super(MaskGate, self).__init__()
        self.gate_cnn = nn.Sequential(
                            nn.Conv2d(512, 256, (1,1), stride=1),
                            nn.ReLU(),
                            nn.Conv2d(256, 128, (1,1), stride=1),
                            nn.ReLU()
        )
        self.gate_fc = nn.Sequential(
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 2),
                            nn.Softmax(dim=-1)
        )
        self.past_num = past_num

    def forward(self,
                param_dict,
                module):

        if module == "Train":
            pred = self.masker_train(param_dict["select_feature"])
            return pred
            
        elif module == "Test":
            m_t, m_d, fill = self.masker_test(
                                        param_dict["all_mask"],
                                        param_dict["present_mask"],
                                        param_dict["feature_bank"],
                                        param_dict["time"],
                                        param_dict["location_all"],
                                        param_dict["masker_dict"])
            return m_t, m_d, fill
        
    def masker_train(self,
                     select_feature):
        
        select_feature_un = select_feature.unsqueeze(-1).unsqueeze(-1)
        temp = self.gate_cnn(select_feature_un)
        temp_sq = temp.squeeze(-1).squeeze(-1)
        pred_status = self.gate_fc(temp_sq)

        return pred_status

    
    def masker_test(self,
                    all_mask,
                    present_mask,
                    temporal_feature_bank,
                    t,
                    location_all,
                    masker_dict):  

        if len(all_mask) == 0:

            return present_mask, masker_dict, (torch.empty(([0,])).type_as(location_all),
                                               torch.empty(([0,])).type_as(location_all))
        else:
            # mask before
            before_mask = all_mask[-1] # shape: batch_size x objects
            # compare from True to False
            index = torch.where((before_mask==True) & (present_mask==False))
            
            if min(index[0].shape) != 0:
                un_index, indexed = self.get_index(t, index, masker_dict, present_mask)
                present_mask[indexed] = True
                temporal_feature_bank_tensor = torch.stack(temporal_feature_bank)[-1,:,:,:]
                
                tfbt_un = temporal_feature_bank_tensor[un_index].unsqueeze(-1).unsqueeze(-1)
                temp = self.gate_cnn(tfbt_un)
                temp_sq = temp.squeeze(-1).squeeze(-1)
                gate_status = self.gate_fc(temp_sq)

                near_feature_infor = location_all[:,t+1:t+self.past_num+1,:,:]                    
                n_index = near_feature_infor[un_index[0],:,un_index[1], :]
                test_label = torch.sum(n_index[:,:,1:5].flatten(start_dim=1), dim=1, keepdim=True).ge(0.01)
            
                shelter_objects = torch.where(gate_status[:, 1] >= 0.50)

                shelter_index = (un_index[0][shelter_objects], un_index[1][shelter_objects])
                present_mask[shelter_index] = True
                fill_position_index = (torch.cat((indexed[0], shelter_index[0])), 
                                        torch.cat((indexed[1], shelter_index[1])))

                training_test_samples = torch.hstack((gate_status, test_label))

                masker_dict['index'][0] = torch.cat((masker_dict['index'][0], un_index[0]))
                masker_dict['index'][1] = torch.cat((masker_dict['index'][1], un_index[1]))
                masker_dict['pred_label'] = torch.cat((masker_dict['pred_label'], training_test_samples))

                for i in range(un_index[0].shape[0]):
                    masker_dict['timestamp'].append(t)

                return present_mask, masker_dict, fill_position_index

            else:

                return present_mask, masker_dict, (torch.empty(([0,])).type_as(location_all),
                                                   torch.empty(([0,])).type_as(location_all))

    def get_index(self,
                  t,
                  present_index,
                  index_dict,
                  present_mask):

        un_index = []
        indexed = []
        for i in range(present_index[0].shape[0]):
            batch_index = torch.where(index_dict['index'][0]==present_index[0][i])
            if min(batch_index[0].shape) != 0:
                object_index = torch.where(index_dict['index'][1][batch_index]==present_index[1][i]) 
                if min(object_index[0].shape) != 0 :
                    # check time
                    nearest_timestamps_index = batch_index[0][object_index[0]][-1] 
                    if (t - index_dict['timestamp'][nearest_timestamps_index] <= self.past_num) & \
                        (present_mask[present_index[0][i].item(), present_index[1][i].item()].item() == False):
                        indexed.append(i)
                    else: 
                        un_index.append(i)  
                else:
                    un_index.append(i)
            else:
                un_index.append(i)
   
        return (present_index[0][un_index], present_index[1][un_index]), \
               (present_index[0][indexed], present_index[1][indexed])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)