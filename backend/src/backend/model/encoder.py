import torch.nn as nn

from backend.model.model_utils import (
    Permute4Batchnorm,
    PositionalEncoding,
    SelfAttLayer_Enc,
    init_xavier_glorot,
)


class Encoder(nn.Module):
    def __init__(self, device, in_feat_dim, time_steps=121, feature_dim=256, head_num=4, k=4):
        super().__init__()
        self.device = device
        self.time_steps = time_steps  # T
        self.feature_dim = feature_dim  # D
        self.head_num = head_num  # H
        # self.max_dynamic_rg = max_dynamic_rg        # GD
        # self.max_static_rg = max_static_rg          # GS
        assert feature_dim % head_num == 0
        self.head_dim = int(feature_dim / head_num)  # d
        self.k = k  # k

        # TODO: replace custom multihead attention layer to nn.MultiHeadAttention
        # layer A : input -> [A,T,in_feat_dim=16] / output -> [A,T,D]
        self.layer_A = nn.Sequential(
            nn.Linear(in_feat_dim, 32),
            Permute4Batchnorm((0, 2, 1)),
            nn.BatchNorm1d(32),
            Permute4Batchnorm((0, 2, 1)),
            nn.ReLU(),
            nn.Linear(32, 128),
            Permute4Batchnorm((0, 2, 1)),
            nn.BatchNorm1d(128),
            Permute4Batchnorm((0, 2, 1)),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            Permute4Batchnorm((0, 2, 1)),
            nn.BatchNorm1d(feature_dim),
            Permute4Batchnorm((0, 2, 1)),
            nn.ReLU(),
        )
        self.layer_A.apply(init_xavier_glorot)

        # Positional embedding
        self.layer_B = PositionalEncoding(self.feature_dim, 0.1, self.time_steps)

        # 10 axis-factorized attention layers alternating between the A and T dimensions for self-attention
        # layer D,E,F,..,Q : input -> [A,T,D] / output -> [A,T,D]
        self.layer_D = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )

        self.layer_E = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )
        self.layer_F = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_G = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )
        self.layer_H = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_I = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )

        # self.layer_J = CrossAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k)
        # self.layer_K = CrossAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k)

        self.layer_L = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_M = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )

        # self.layer_N = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True)
        # self.layer_O = SelfAttLayer_Enc(self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False)

        self.layer_P = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=True
        )
        self.layer_Q = SelfAttLayer_Enc(
            self.time_steps, self.feature_dim, self.head_num, self.k, across_time=False
        )

        # self.layer_DH = nn.Sequential(nn.Linear(self.feature_dim,self.feature_dim*4), Permute4Batchnorm((0,2,1)),
        # nn.BatchNorm1d(self.feature_dim*4), Permute4Batchnorm((0,2,1)), nn.ReLU())
        # self.layer_DH.apply(init_xavier_glorot)

    def forward(self, state_feat, agent_batch_mask, padding_mask, hidden_mask, agent_ids_batch):
        # print(state_feat.shape)
        state_feat = state_feat.clone()
        state_feat[hidden_mask] = -1

        A_ = self.layer_A(state_feat)

        output = self.layer_B(A_)
        output = self.layer_D(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_E(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_F(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_G(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_H(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_I(output, agent_batch_mask, padding_mask, hidden_mask)

        output = self.layer_L(output, agent_batch_mask, padding_mask, hidden_mask)
        output = self.layer_M(output, agent_batch_mask, padding_mask, hidden_mask)

        # output = self.layer_N(output,agent_batch_mask, padding_mask, hidden_mask)
        # output = self.layer_O(output,agent_batch_mask, padding_mask, hidden_mask)

        output = self.layer_P(output, agent_batch_mask, padding_mask, hidden_mask)
        Q_ = self.layer_Q(output, agent_batch_mask, padding_mask, hidden_mask)
        # Q_ = self.layer_DH(Q_)
        return {"out": Q_, "att_weights": 0}
