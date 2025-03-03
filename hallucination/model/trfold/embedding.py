import torch
import torch.nn as nn
import torch.nn.functional as F

from hallucination.model.trfold.axialnet import Pair2PairBlock


class Template_Embedding(nn.Module):
    def __init__(self, template_channels=64, groups=4):
        super(Template_Embedding, self).__init__()
        self.channels = template_channels
        self.axialblock_0 = Pair2PairBlock(
            channels=self.channels, groups=groups, ff_ratio=2
        )
        self.axialblock_1 = Pair2PairBlock(
            channels=self.channels, groups=groups, ff_ratio=2
        )
        self.norm = nn.LayerNorm(template_channels)

    def forward(self, x):
        N, T, L, L, C = x.shape
        x = torch.reshape(x, (N * T, L, L, C))
        x = self.axialblock_0(x)
        x = self.axialblock_1(x)
        x = torch.reshape(x, (N, T, L, L, self.channels))
        x = self.norm(x)
        return x


class PointwiseAttn(nn.Module):
    def __init__(self, q_channels=256, kv_channels=64, groups=4):
        super(PointwiseAttn, self).__init__()
        assert kv_channels % groups == 0
        self.groups = groups
        self.ppg = kv_channels // groups
        self.q_channels = q_channels
        self.kv_channels = kv_channels
        self.toQ = nn.Linear(q_channels, kv_channels, bias=False)
        self.toK = nn.Linear(kv_channels, kv_channels, bias=False)
        self.toV = nn.Linear(kv_channels, kv_channels, bias=False)
        self.toOut = nn.Linear(kv_channels, q_channels)

    def forward(self, q, kv):
        N, L, L, C = q.shape
        T = kv.shape[1]
        q = self.toQ(q).reshape(N, L, L, self.groups, self.ppg)
        k = self.toK(kv).reshape(N, T, L, L, self.groups, self.ppg)
        v = self.toV(kv).reshape(N, T, L, L, self.groups, self.ppg)
        attn = torch.einsum("nijgc,ntijgc->ntijg", q, k / self.ppg**0.5)
        attn = F.softmax(attn, dim=1)
        x = torch.einsum("ntijg,ntijgc->nijgc", attn, v).reshape(
            N, L, L, self.kv_channels
        )
        x = self.toOut(x)
        return x


class TR_embedding(nn.Module):
    EMBEDDING_MODEL_TYPE_24BLOCK = "24blocks"
    EMBEDDING_MODEL_TYPE_48BLOCK = "48blocks"

    def __init__(
        self,
        MSA_channels=256,
        pair_channels=256,
        template_channels=64,
        vol_size=22,
        model_type=EMBEDDING_MODEL_TYPE_24BLOCK,
    ):
        assert model_type in [
            self.EMBEDDING_MODEL_TYPE_24BLOCK,
            self.EMBEDDING_MODEL_TYPE_48BLOCK,
        ]
        super(TR_embedding, self).__init__()
        self.MSA_channels = MSA_channels
        self.pair_channels = pair_channels
        self.vol_size = vol_size
        #        self.template_proj=nn.Linear(82,template_channels)

        self.template_attn = PointwiseAttn(pair_channels, template_channels, 4)
        self.msa_embedding_layer = nn.Linear(2 * vol_size, MSA_channels)
        self.query_embedding_l = nn.Linear(vol_size, pair_channels)
        self.query_embedding_r = nn.Linear(vol_size, pair_channels)
        self.query_embedding_m = nn.Linear(vol_size, MSA_channels)
        self.model_type = model_type
        if model_type == self.EMBEDDING_MODEL_TYPE_48BLOCK:
            self.PE_embedding_layer = nn.Embedding(
                num_embeddings=65, embedding_dim=pair_channels
            )
        elif model_type == self.EMBEDDING_MODEL_TYPE_24BLOCK:
            self.PE_embedding_layer = nn.Embedding(
                num_embeddings=33, embedding_dim=pair_channels
            )
        self.recycle_query_norm = nn.LayerNorm(MSA_channels)
        self.recycle_rr_norm = nn.LayerNorm(pair_channels)
        self.recycle_dist_proj = nn.Linear(15, pair_channels)

    def forward(
        self, msa_matrix, template, idx, recycle_query, recycle_rr, recycle_dist
    ):
        recycle_dist = recycle_dist.to(recycle_rr.dtype)
        N, D, L, _ = msa_matrix.shape
        seq = msa_matrix[:, 0, :, :22].clone()
        recycle_query = self.recycle_query_norm(recycle_query)
        recycle_rr = self.recycle_rr_norm(recycle_rr) + self.recycle_dist_proj(
            recycle_dist
        )
        msa_matrix = self.msa_embedding_layer(msa_matrix) + self.query_embedding_m(
            seq
        ).unsqueeze(1)
        msa_matrix[:, 0, :, :] = msa_matrix[:, 0, :, :] + recycle_query
        rr_edge = (
            self.query_embedding_l(seq)[:, :, None, :]
            + self.query_embedding_r(seq)[:, None, :, :]
        )
        #        template=self.template_proj(template)
        #        template=self.template_embedding_layer(template,idx)
        if self.model_type == self.EMBEDDING_MODEL_TYPE_24BLOCK:
            seqsep = torch.clip(
                torch.abs(idx[:, :, None] - idx[:, None, :]), min=0, max=32
            )
        elif self.model_type == self.EMBEDDING_MODEL_TYPE_48BLOCK:
            seqsep = torch.clip(idx[:, :, None] - idx[:, None, :], min=-32, max=32) + 32

        pe2d = self.PE_embedding_layer(seqsep)
        rr_edge = rr_edge + pe2d + recycle_rr
        rr_edge = rr_edge + self.template_attn(rr_edge, template)
        return msa_matrix, rr_edge
