import torch
import torch.nn as nn
import torch.nn.functional as F


from ..builder import ATTNMODEL


@ATTNMODEL.register_module()
class STAttn(nn.Module):

    def __init__(self):

        super(STAttn, self).__init__()

        self.ue = nn.Linear(512, 64)
        self.be = nn.Parameter(torch.zeros(64))
        self.w = nn.Linear(64, 1)

        self.fc1 = nn.Linear(512, 256)

        self.softmax = nn.Softmax(dim=-2)

    def forward(self,
                inputs,
                module): 

        if module=="Train":
            fc_attr = self.attn_train(inputs)
        elif module=="Test":
            fc_attr = self.attn_test(inputs)
            
        return fc_attr

    def attn_train(self, inputs):

        inputs_flatten = torch.flatten(inputs, start_dim=0, end_dim=1)
        # weight the feature
        e_j = self.w(F.leaky_relu((self.ue(inputs_flatten) + self.be), negative_slope=0.2))
        a_j = self.softmax(e_j)
        attr = torch.mul(a_j, inputs_flatten).sum(dim=1)
        fc_attr = self.fc1(attr)
        fc_attr_re = fc_attr.reshape(inputs.shape[0], inputs.shape[1], 256)

        fc_attr_re_lstm = fc_attr_re.permute(1,0,2)

        return fc_attr_re_lstm

    def attn_test(self, inputs):

        # mean the spatial_feature
        e_j = self.w(F.leaky_relu((self.ue(inputs) + self.be), negative_slope=0.2))
        a_j = self.softmax(e_j)
        attr = torch.mul(a_j, inputs).sum(dim=1)
        # counting down the dementions
        fc_attr = self.fc1(attr)

        return fc_attr 

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
