import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallucination.utils import AxialDropout, FeedForwardLayer


class TriangMul(nn.Module):
    def __init__(self, channels, outgoing=True):
        super(TriangMul, self).__init__()
        self.ln = nn.LayerNorm(channels)
        self.linear_l1 = nn.Linear(channels, channels)
        self.linear_l2 = nn.Linear(channels, channels)
        self.linear_r1 = nn.Linear(channels, channels)
        self.linear_r2 = nn.Linear(channels, channels)
        self.linear_gate = nn.Linear(channels, channels)
        self.linear_out = nn.Linear(channels, channels)
        self.outgoing = outgoing

        nn.init.constant_(self.linear_gate.weight.data, 0)
        nn.init.constant_(self.linear_gate.bias.data, 1)
        nn.init.constant_(self.linear_out.weight.data, 0)

    def forward(self, x):
        left = torch.sigmoid(self.linear_l1(x)) * self.linear_l2(x)
        right = torch.sigmoid(self.linear_r1(x)) * self.linear_r2(x)
        if self.outgoing:
            out = self.ln(torch.einsum("nikc,njkc->nijc", left, right))
        else:
            out = self.ln(torch.einsum("nkic,nkjc->nijc", left, right))
        out = torch.sigmoid(self.linear_gate(out)) * self.linear_out(out)
        return out


class TriangAxial(nn.Module):
    def __init__(self, channels=128, groups=4, outgoing=True):
        super(TriangAxial, self).__init__()
        assert channels % groups == 0
        self.to_Query = nn.Linear(channels, channels, bias=False)
        self.to_Key = nn.Linear(channels, channels, bias=False)
        self.to_Value = nn.Linear(channels, channels, bias=False)
        self.attn_bias = nn.Linear(channels, groups, bias=False)
        self.gate = nn.Linear(channels, channels)
        self.to_Out = nn.Linear(channels, channels)
        self.outgoing = outgoing
        self.groups = groups
        self.channels = channels
        self.ppg = channels // groups

        nn.init.constant_(self.gate.weight.data, 0)
        nn.init.constant_(self.gate.bias.data, 1)
        nn.init.constant_(self.to_Out.weight.data, 0)

    def forward(self, x):
        N, L, L, C = x.shape
        if not self.outgoing:
            x = x.permute(0, 2, 1, 3)
        gate = torch.sigmoid(self.gate(x)).reshape(N, L, L, self.groups, self.ppg)
        Query = (
            self.to_Query(x).reshape(N, L, L, self.groups, self.ppg) / self.ppg**0.5
        )
        Key = self.to_Key(x).reshape(N, L, L, self.groups, self.ppg)
        Value = self.to_Value(x).reshape(N, L, L, self.groups, self.ppg)
        attn = torch.einsum("nijgc,nikgc->nijkg", Query, Key) + self.attn_bias(
            x
        ).unsqueeze(1)
        attn = F.softmax(attn, dim=2)
        x = gate * torch.einsum("nijkg,nikgc->nijgc", attn, Value)
        x = x.reshape(N, L, L, C)
        x = self.to_Out(x)
        if not self.outgoing:
            x = x.permute(0, 2, 1, 3)
        return x


class AxialAttention(nn.Module):
    def __init__(self, channels, groups=8, width=False, p_drop=0.1):
        assert channels % groups == 0
        super(AxialAttention, self).__init__()
        self.channels = channels
        self.groups = groups
        self.group_planes = channels // groups
        self.width = width

        # Multi-head self attention
        self.conv1x1_Q = nn.Linear(channels, channels, bias=False)
        self.conv1x1_K = nn.Linear(channels, channels, bias=False)
        self.conv1x1_V = nn.Linear(channels, channels, bias=False)
        self.conv1x1_O = nn.Linear(channels, channels)

    def forward(self, x):
        if not self.width:
            x = x.permute(0, 2, 1, 3)  # N, W, C, H
        N, W, H, C = x.shape
        x = torch.reshape(x, (N * W, H, C))
        # Transformations
        q = self.conv1x1_Q(x).reshape(N * W, H, self.groups, self.group_planes)
        k = self.conv1x1_K(x).reshape(N * W, H, self.groups, self.group_planes)
        v = self.conv1x1_V(x).reshape(N * W, H, self.groups, self.group_planes)
        attn = torch.einsum("nigc, njgc->nijg", q, k / math.sqrt(self.group_planes))
        attn = F.softmax(attn, dim=2)
        output = torch.einsum("nijg,njgc->nigc", attn, v)
        output = torch.reshape(output, (N * W, H, self.channels))
        output = self.conv1x1_O(output)
        output = torch.reshape(output, (N, W, H, self.channels))
        if not self.width:
            output = output.permute(0, 2, 1, 3).contiguous()
        return output


class Pair2PairBlock(nn.Module):
    def __init__(self, channels=256, groups=8, p_drop=0.25, ff_ratio=4):
        super(Pair2PairBlock, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        #        self.hight_block = AxialAttention(channels, groups=groups,p_drop=p_drop)
        #        self.width_block = AxialAttention(channels, groups=groups, width=True,p_drop=p_drop)
        self.triangmul_out = TriangMul(channels, outgoing=True)
        self.triangmul_in = TriangMul(channels, outgoing=False)
        self.triangattn_out = TriangAxial(channels, outgoing=True)
        self.triangattn_in = TriangAxial(channels, outgoing=False)
        self.ff = FeedForwardLayer(channels, ff_ratio * channels)

        self.dropout1 = AxialDropout(p_drop, orientation="row")
        self.dropout2 = AxialDropout(p_drop, orientation="row")
        self.dropout3 = AxialDropout(p_drop, orientation="row")
        self.dropout4 = AxialDropout(p_drop, orientation="col")

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.norm4 = nn.LayerNorm(channels)

    def forward(self, x):
        raw = x
        x = self.norm1(x)
        x = self.triangmul_out(x)
        x = raw + self.dropout1(x)

        raw = x
        x = self.norm2(x)
        x = self.triangmul_in(x)
        x = raw + self.dropout2(x)

        raw = x
        x = self.norm3(x)
        x = self.triangattn_out(x)
        x = raw + self.dropout3(x)

        raw = x
        x = self.norm4(x)
        x = self.triangattn_in(x)
        x = raw + self.dropout4(x)

        x = self.ff(x)
        return x


class MSARowAttention(nn.Module):
    def __init__(self, channels=256, bias_channels=256, groups=8, p_drop=0.15):
        super(MSARowAttention, self).__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.ppg = channels // groups
        self.Key = nn.Linear(channels, channels, bias=False)
        self.Query = nn.Linear(channels, channels, bias=False)
        self.Value = nn.Linear(channels, channels, bias=False)
        self.Bias = nn.Linear(bias_channels, groups, bias=False)
        self.Out = nn.Linear(channels, channels)
        self.Gate = nn.Linear(channels, channels)
        self.drop = AxialDropout(p_drop, orientation="row")
        self.norm = nn.LayerNorm(channels)
        self.norm_bias = nn.LayerNorm(bias_channels)

        nn.init.constant_(self.Gate.weight.data, 0)
        nn.init.constant_(self.Gate.bias.data, 1)
        nn.init.constant_(self.Out.weight.data, 0)

    def forward(self, x, bias):
        N, D, L, C = x.shape
        bias = self.norm_bias(bias)
        bias = self.Bias(bias)
        raw = x
        x = self.norm(x)
        Gate = torch.sigmoid(self.Gate(x)).reshape(N, D, L, self.ppg, self.groups)
        Key = torch.reshape(self.Key(x), (N, D, L, self.ppg, self.groups)) / math.sqrt(
            self.ppg
        )
        Query = torch.reshape(self.Query(x), (N, D, L, self.ppg, self.groups)) / D
        x = torch.reshape(self.Value(x), (N, D, L, self.ppg, self.groups))
        attn = torch.einsum("ndlcg,ndmcg->nlmg", Query, Key)
        attn = attn + bias
        attn = F.softmax(attn, dim=2)
        x = Gate * torch.einsum("nlmg,ndmcg->ndlcg", attn, x)
        x = torch.reshape(x, (N, D, L, C))
        x = self.Out(x)
        x = self.drop(x) + raw
        return x


class MSAColAttention(nn.Module):
    def __init__(self, channels=256, groups=8, p_drop=0.1):
        super(MSAColAttention, self).__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.ppg = channels // groups
        self.Key = nn.Linear(channels, channels, bias=False)
        self.Query = nn.Linear(channels, channels, bias=False)
        self.Value = nn.Linear(channels, channels, bias=False)
        self.Out = nn.Linear(channels, channels)
        self.Gate = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        nn.init.constant_(self.Gate.weight.data, 0)
        nn.init.constant_(self.Gate.bias.data, 1)
        nn.init.constant_(self.Out.weight.data, 0)

    def forward(self, x):
        N, D, L, C = x.shape
        raw = x
        x = self.norm(x)
        Gate = torch.sigmoid(self.Gate(x)).reshape(N, D, L, self.ppg, self.groups)
        Key = torch.reshape(self.Key(x), (N, D, L, self.ppg, self.groups)) / math.sqrt(
            self.ppg
        )
        Query = torch.reshape(self.Query(x), (N, D, L, self.ppg, self.groups))
        x = torch.reshape(self.Value(x), (N, D, L, self.ppg, self.groups))
        attn = torch.einsum("ndlcg,nklcg->ndklg", Query, Key)
        attn = F.softmax(attn, dim=2)
        x = Gate * torch.einsum("ndklg,nklcg->ndlcg", attn, x)
        x = torch.reshape(x, (N, D, L, C))
        x = self.Out(x)
        x = x + raw
        return x


class MSAColGlobalAttention(nn.Module):
    def __init__(self, channels=256, groups=8, p_drop=0.1):
        super(MSAColGlobalAttention, self).__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.ppg = channels // groups
        self.Key = nn.Linear(channels, channels, bias=False)
        self.Query = nn.Linear(channels, channels, bias=False)
        self.Value = nn.Linear(channels, channels, bias=False)
        self.Out = nn.Linear(channels, channels)
        self.Gate = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        nn.init.constant_(self.Gate.weight.data, 0)
        nn.init.constant_(self.Gate.bias.data, 1)
        nn.init.constant_(self.Out.weight.data, 0)

    def forward(self, x):
        N, D, L, C = x.shape
        raw = x
        x = self.norm(x)
        Key = torch.reshape(self.Key(x), (N, D, L, self.ppg, self.groups)) / math.sqrt(
            self.ppg
        )
        Query = torch.reshape(self.Query(x.mean(dim=1)), (N, L, self.ppg, self.groups))
        Value = torch.reshape(self.Value(x), (N, D, L, self.ppg, self.groups))
        attn = torch.einsum("nlcg,nklcg->nklg", Query, Key)
        attn = F.softmax(attn, dim=1)
        Gate = torch.sigmoid(self.Gate(x)).reshape(N, D, L, self.ppg, self.groups)
        x = Gate * torch.einsum("nklg,nklcg->nlcg", attn, Value).unsqueeze(1)
        x = torch.reshape(x, (N, D, L, C))
        x = self.Out(x)
        x = x + raw
        return x


class MSA_Attn(nn.Module):
    def __init__(self, channels=256, bias_channels=256, groups=8, Extra=False):
        super(MSA_Attn, self).__init__()
        self.channels = channels
        self.groups = groups
        self.group_planes = channels // groups
        d = self.group_planes
        self.msa_row_attn = MSARowAttention(
            channels=channels, bias_channels=bias_channels, groups=groups
        )
        if Extra:
            self.msa_col_attn = MSAColGlobalAttention(channels=channels, groups=groups)
        else:
            self.msa_col_attn = MSAColAttention(channels=channels, groups=groups)
        self.ff = FeedForwardLayer(channels, 4 * channels)

    def forward(self, x, bias):
        N, D, L, C = x.shape
        x = self.msa_row_attn(x, bias)
        x = self.msa_col_attn(x)
        x = self.ff(x)
        return x
