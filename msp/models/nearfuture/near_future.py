import torch
import torch.nn as nn

from ..builder import NEARFUTUREMODEL


@NEARFUTUREMODEL.register_module()
class NearFuture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                location_all, 
                feature_all,
                t,
                fill_position_index):
        if t==0:
            return location_all[:, t], feature_all[:, t]
        else:
        # process for location prediction
            if min(fill_position_index[0].shape) != 0:
                if t <= 5:
                    past_movement_bank = location_all[fill_position_index[0], :t, fill_position_index[1], :].data
                else:
                    past_movement_bank = location_all[fill_position_index[0], (t-5):t, fill_position_index[1], :].data

                real_timestamp = past_movement_bank.shape[1]
                delta_movement = (past_movement_bank[:,-1] - past_movement_bank[:, 0])/(real_timestamp -1)
                pred_location = past_movement_bank[:, -1, 1:5] - delta_movement[:, 1:5]
                location_all[fill_position_index[0], t, fill_position_index[1], 1:5].data = pred_location

                # fill the feature
                past_feature = feature_all[fill_position_index[0], t-1, fill_position_index[1], :].data
                feature_all[fill_position_index[0], t, fill_position_index[1], :].data = past_feature
                return location_all[:, t], feature_all[:, t]

            else:
                return location_all[:, t], feature_all[:, t]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)