import torch
import torch.nn as nn
import torch.nn.functional as F

from hallucination.utils import FeedForwardLayer


class Update_Edge(nn.Module):
    def __init__(self, MSA_channels=256, pair_channels=256, outerprod_channels=32):
        super(Update_Edge, self).__init__()
        self.outerprod_channels = outerprod_channels
        self.MSA_channels = MSA_channels
        self.pair_channels = pair_channels
        #        self.attn_channels=attn_channels

        self.outerprod_l = nn.Linear(MSA_channels, outerprod_channels)
        self.outerprod_r = nn.Linear(MSA_channels, outerprod_channels)
        self.Out = nn.Linear(outerprod_channels**2, pair_channels)
        self.norm_msa = nn.LayerNorm(MSA_channels)

        nn.init.constant_(self.Out.weight.data, 0)

    def forward(self, msa_matrix):
        N, D, L, C = msa_matrix.shape
        msa_matrix = self.norm_msa(msa_matrix)
        outprod = torch.einsum(
            "ndlc,ndmq->nlmcq",
            self.outerprod_l(msa_matrix),
            self.outerprod_r(msa_matrix) / D,
        )
        outprod = torch.reshape(outprod, (N, L, L, self.outerprod_channels**2))
        outprod = self.Out(outprod)
        return outprod


class Update_Node(nn.Module):
    def __init__(self, MSA_channels=256, pair_channels=256, groups=4, p_drop=0.1):
        super(Update_Node, self).__init__()
        self.MSA_channels = MSA_channels
        self.pair_channels = pair_channels
        self.groups = groups
        self.group_planes = MSA_channels // groups

        self.V = nn.Linear(MSA_channels, MSA_channels, kernel_size=1)
        self.A = nn.Linear(pair_channels, groups, kernel_size=1)
        self.O = nn.Linear(MSA_channels, MSA_channels, kernel_size=1)

        self.ff = FeedForwardLayer(MSA_channels, 4 * MSA_channels)
        self.drop_1 = nn.Dropout(p_drop, inplace=False)
        self.drop_2 = nn.Dropout(p_drop, inplace=True)
        self.norm_msa = nn.LayerNorm(MSA_channels)
        self.norm_rr = nn.LayerNorm(pair_channels)

    def forward(self, msa_matrix, rr_edge):
        N, D, L, C = msa_matrix.shape
        raw = msa_matrix
        msa_matrix = self.norm_msa(msa_matrix)
        rr_edge = 0.5 * (rr_edge + rr_edge.transpose(1, 2))
        E = self.norm_rr(rr_edge)
        A = self.A(E)
        A = F.softmax(A, dim=2)
        A = self.drop_1(A)
        V = torch.reshape(
            self.conv1x1_V(msa_matrix), (N, D, L, self.groups, self.group_planes)
        )

        msa_matrix = torch.einsum("nijg,ndjgc->ndigc", A, V)
        msa_matrix = torch.reshape(msa_matrix, (N, D, L, C))
        msa_matrix = self.conv1x1_O(msa_matrix)
        msa_matrix = raw + self.drop_2(msa_matrix)

        msa_matrix = self.ff(msa_matrix)
        return msa_matrix
