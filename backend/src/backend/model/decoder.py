import torch
import torch.nn.functional as FU
from torch import nn

from backend.model.model_utils import Permute4Batchnorm, SelfAttLayer_Dec, init_xavier_glorot


class Decoder(nn.Module):
    def __init__(self, device, time_steps=121, feature_dim=256, head_num=4, k=4, F=6):
        super().__init__()
        self.device = device
        self.time_steps = time_steps  # T
        self.feature_dim = feature_dim  # D
        self.head_num = head_num  # H
        self.k = k
        self.F = F

        onehots_ = torch.tensor(range(F))
        self.onehots_ = FU.one_hot(onehots_, num_classes=F).to(self.device)

        self.layer_T = nn.Sequential(nn.Linear(self.feature_dim + self.F, feature_dim), nn.ReLU())
        # self.layer_T.apply(init_xavier_glorot)

        self.layer_U = SelfAttLayer_Dec(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_V = SelfAttLayer_Dec(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )

        self.layer_W = SelfAttLayer_Dec(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_X = SelfAttLayer_Dec(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )

        self.layer_Y = nn.LayerNorm(self.feature_dim)

        self.layer_Z1 = nn.Sequential(
            nn.Linear(self.feature_dim, 4),
            nn.ReLU(),
            Permute4Batchnorm((1, 3, 0, 2)),
            nn.BatchNorm2d(4),
            nn.Softplus(),
            Permute4Batchnorm((2, 0, 3, 1)),
        )  # The extra softplus at the end is to ensure that the output parameters are all > 0
        # Output for x,y laplace distribution parameters 4 x.loc,x.scale,y.loc,y.scale
        self.layer_Z1.apply(init_xavier_glorot)

    def forward(self, state_feat, batch_mask, padding_mask, hidden_mask=None):
        A, T, D = state_feat.shape
        assert self.time_steps == T and self.feature_dim == D
        # state_feat = state_feat.reshape((A,T,-1,self.F))
        # x = state_feat.permute(3,0,1,2)

        onehots_ = self.onehots_.view(self.F, 1, 1, self.F).repeat(1, A, T, 1)
        onehots_ = onehots_.to(state_feat.device)
        x = state_feat.unsqueeze(0).repeat(self.F, 1, 1, 1)

        x = torch.cat((x, onehots_), dim=-1)
        x = self.layer_T(x)

        x = self.layer_U(x, batch_mask=batch_mask)  # , padding_mask=padding_mask)
        x = self.layer_V(x, batch_mask=batch_mask)  # , padding_mask=padding_mask)

        x = self.layer_W(x, batch_mask=batch_mask)  # , padding_mask=padding_mask)
        x = self.layer_X(x, batch_mask=batch_mask)  # , padding_mask=padding_mask)

        x = self.layer_Y(x)
        x = self.layer_Z1(x)
        # x = self.layer_Z2(x)                                # [F,A,T,D]

        return x
