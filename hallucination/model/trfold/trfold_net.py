import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hallucination.model.trfold.axialnet import MSA_Attn, Pair2PairBlock
from hallucination.model.trfold.embedding import TR_embedding, Template_Embedding
from hallucination.model.trfold.structure import StructureModule
from hallucination.model.trfold.update_net import Update_Edge
from hallucination.utils import coord2bindist, symmetry


class TR_block(nn.Module):
    def __init__(self, MSA_channels=256, pair_channels=256, Extra=False):
        super(TR_block, self).__init__()
        self.MSA_channels = MSA_channels
        self.pair_channels = pair_channels
        self.msa_attn = MSA_Attn(
            channels=MSA_channels, bias_channels=pair_channels, groups=8, Extra=Extra
        )
        self.axialblock = Pair2PairBlock(channels=pair_channels, groups=4)
        self.update_edge = Update_Edge(
            MSA_channels=MSA_channels, pair_channels=pair_channels
        )

    def forward(self, msa_matrix, rr_edge):
        msa_matrix = self.msa_attn(msa_matrix, rr_edge)
        rr_edge = rr_edge + self.update_edge(msa_matrix)
        msa_matrix = msa_matrix.cpu()
        rr_edge = self.axialblock(rr_edge)
        msa_matrix = msa_matrix.cuda()
        return msa_matrix, rr_edge


class OutputLayer(nn.Module):
    def __init__(self, MSA_channels=256, pair_channels=256):
        super(OutputLayer, self).__init__()
        self.pair_channels = pair_channels
        self.MSA_channels = MSA_channels
        self.to_dist = nn.Linear(pair_channels, 64)
        self.to_msa = nn.Linear(MSA_channels, 21)
        self.to_ptm = nn.Linear(pair_channels, 64)
        nn.init.constant_(self.to_dist.weight.data, 0)
        nn.init.constant_(self.to_msa.weight.data, 0)
        nn.init.constant_(self.to_ptm.weight.data, 0)

    def forward(self, msa, rr_edge):
        msa = self.to_msa(msa)  # logits
        msa = msa.cpu()
        ptm = self.to_ptm(rr_edge)  # logits
        ptm = ptm.cpu()
        rr_edge = symmetry(rr_edge)
        distance = self.to_dist(rr_edge)  # logits

        msa = msa.cuda()
        ptm = ptm.cuda()
        return distance, msa, ptm


class TRFold(nn.Module):
    def __init__(
        self,
        blocks=1,
        MSA_channels=256,
        pair_channels=256,
        device="cpu",
        embedding_model_type=TR_embedding.EMBEDDING_MODEL_TYPE_24BLOCK,
    ):
        super().__init__()
        # setup logger

        self.blocks = blocks
        self.MSA_channels = MSA_channels
        self.pair_channels = pair_channels

        self.template_proj = nn.Linear(82, 64)
        self.template_embedding_layer = Template_Embedding(64, 4)

        self.embedding_layer = TR_embedding(
            MSA_channels=MSA_channels,
            pair_channels=pair_channels,
            vol_size=22,
            model_type=embedding_model_type,
        )
        self.extra_msa_embedding = nn.Embedding(num_embeddings=21, embedding_dim=64)
        self.extra_blocks = nn.ModuleList(
            [
                TR_block(MSA_channels=64, pair_channels=pair_channels, Extra=True)
                for i in range(4)
            ]
        )
        self.af2blocks = nn.ModuleList(
            [
                TR_block(MSA_channels=MSA_channels, pair_channels=pair_channels)
                for i in range(blocks)
            ]
        )
        self.output = OutputLayer(MSA_channels, pair_channels)
        self.proj_node = nn.Linear(MSA_channels, 384)
        self.structure = StructureModule(device=device)
        self.recycle_query = None
        self.recycle_rr = None
        self.recycle_dist = None

    def forward(self, msa_matrix, extra_msa, template, idx, aatype, bf16=True):
        template = self.template_proj(template)
        template = self.template_embedding_layer(template)
        msa_matrix, rr_edge = checkpoint(
            self.embedding_layer,
            msa_matrix,
            template,
            idx,
            self.recycle_query,
            self.recycle_rr,
            self.recycle_dist,
        )
        # msa_matrix = msa_matrix.cpu()
        extra_msa = self.extra_msa_embedding(extra_msa)
        for i in range(4):
            extra_msa, rr_edge = checkpoint(self.extra_blocks[i], extra_msa, rr_edge)
        del extra_msa
        # msa_matrix = msa_matrix.cuda()
        for i in range(self.blocks):
            msa_matrix, rr_edge = checkpoint(self.af2blocks[i], msa_matrix, rr_edge)
        self.recycle_query = msa_matrix[:, 0, :, :].clone().detach()
        self.recycle_rr = rr_edge.detach()
        #        return self.output(msa_matrix,rr_edge)
        distance, msa, ptm = self.output(msa_matrix, rr_edge)
        distance = distance.cpu()
        msa = msa.cpu()
        ptm = ptm.cpu()
        node = msa_matrix[:, 0, :, :].clone()
        del msa_matrix

        node = self.proj_node(node)
        bb_frame_traj, torsion_traj, frames, atoms, atom_mask, plddt = self.structure(
            node, rr_edge, aatype
        )
        plddt = plddt.cpu()
        atom_mask = atom_mask.cpu()
        recycle_cb = atoms[:, :, 4, :].detach()
        atoms = atoms.cpu()
        self.recycle_dist = F.one_hot(
            coord2bindist(recycle_cb, min=3.375, width=1.25, bins=15),
            num_classes=15,
        )
        if bf16:
            self.recycle_dist = self.recycle_dist.bfloat16()
        # return everything on cpu
        res = {
            "distance": distance.float(),
            "msa": msa.float(),
            "bb_frame_traj": bb_frame_traj,
            "torsion_traj": torsion_traj,
            "frames": frames,
            "atoms": atoms.float(),
            "atom_mask": atom_mask.float(),
            "plddt": plddt.float(),
            "ptm": ptm.float(),
        }
        return res
