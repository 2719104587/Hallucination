import math
import os
import pickle
import string
from Bio.PDB import PDBParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from hallucination.model.alphafold2.residue_constants import (
    restype_1to3,
    restype_name_to_atom14_names,
    atom_types,
    restype_3to1,
)

# read A3M and convert letters into
# integers in the 0..20 range
alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
aa_1_N = {a: n for n, a in enumerate(alpha_1)}


def id2aa(idx):
    return alpha_1[idx]


def write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_fasta(query_fasta):
    fp = open(query_fasta, "r")
    lines = fp.read()
    fp.close()
    sequence = ""
    filename = os.path.basename(query_fasta)
    query_id = os.path.splitext(filename)[0]
    for line in lines.splitlines():
        if not line.startswith(">"):
            sequence += line.strip()
    sequence = sequence.upper().replace(" ", "").replace("\t", "")
    return query_id, sequence


def read_pdb(pdb_path, motif_loc, motif_loc_chain, predict_model):
    parser = PDBParser(PERMISSIVE=True)
    template_struct = parser.get_structure("X", pdb_path)
    chains = {chain.id: chain for chain in template_struct.get_chains()}
    residues = list(chains[motif_loc_chain].get_residues())

    assert len(motif_loc) == 2

    coord_list = []
    atom_mask_list = []
    seq = ""
    motif_residue_id = [int(i) for i in range(motif_loc[0], motif_loc[1] + 1)]
    for residue in residues:
        residue_id = int(residue.id[1])
        if residue_id in motif_residue_id:
            residue_name = residue.get_resname()
            seq += restype_3to1[residue_name]

            atom_coord = torch.zeros((1, 1, 37, 3))
            atom_mask = torch.zeros((1, 1, 37))
            atoms = list(residue.get_atoms())
            for atom in atoms:
                atom_index = atom_types.index(atom.name)
                atom_mask[:, :, atom_index] = 1
                atom_coord[:, :, atom_index, 0] = atom.get_vector()[0]
                atom_coord[:, :, atom_index, 1] = atom.get_vector()[1]
                atom_coord[:, :, atom_index, 2] = atom.get_vector()[2]
            coord_list.append(atom_coord)
            atom_mask_list.append(atom_mask)

    if predict_model == "trfold":
        return seq, torch.cat(coord_list, dim=1), None
    elif predict_model == "alphafold":
        return (
            seq,
            torch.cat(coord_list, dim=1).numpy(),
            torch.cat(atom_mask_list, dim=1).numpy(),
        )


def to_pdb(sequence, coords, plddt, res_mask, chain_info, predict_model):
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    info_dict = {}
    length = 0
    for i, info in enumerate(chain_info):
        chain_length = info[1]
        for chain_res, res_index in enumerate(range(length, length + chain_length)):
            info_dict[res_index] = [chain_ids[i], chain_res]
        length += chain_length

    assert length == len(sequence)

    pdb_str = ""
    atom_index = 1
    for res in range(len(sequence)):
        #### stick to 20 standard amino acid types
        resname = restype_1to3.get(sequence[res])
        chainID = info_dict[res][0]
        resSeq = f"{info_dict[res][1] + 1:>4}"
        iCode = "    "
        bfactor = "{:.2f}".format(plddt[res])

        if predict_model == "trfold":
            for atom in range(14):
                xcord = coords[res, atom, 0]
                ycord = coords[res, atom, 1]
                zcord = coords[res, atom, 2]
                x = f"{xcord:>8.3f}"
                y = f"{ycord:>8.3f}"
                z = f"{zcord:>8.3f}"
                atom_name = restype_name_to_atom14_names[resname][atom]
                if atom_name == "":
                    continue

                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                element = atom_name[0]
                serial = f"{atom_index:>5}"
                occupancy = "1.00"
                line = (
                    "ATOM  "
                    + serial
                    + " "
                    + f"{name:<4}"
                    + " "
                    + resname
                    + " "
                    + chainID
                    + resSeq
                    + iCode
                    + x
                    + y
                    + z
                    + f"{occupancy:>6}"
                    + f"{bfactor:>6}"
                    + "           "
                    + element
                )
                pdb_str += line
                pdb_str += "\n"
                atom_index += 1
        elif predict_model == "alphafold":
            for atom in range(37):
                if res_mask[res, atom] >= 0.5:
                    xcord = coords[res, atom, 0]
                    ycord = coords[res, atom, 1]
                    zcord = coords[res, atom, 2]
                    x = f"{xcord:>8.3f}"
                    y = f"{ycord:>8.3f}"
                    z = f"{zcord:>8.3f}"
                    atom_name = atom_types[atom]
                    if atom_name == "":
                        continue

                    name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                    element = atom_name[0]
                    serial = f"{atom_index:>5}"
                    occupancy = "1.00"
                    line = (
                        "ATOM  "
                        + serial
                        + " "
                        + f"{name:<4}"
                        + " "
                        + resname
                        + " "
                        + chainID
                        + resSeq
                        + iCode
                        + x
                        + y
                        + z
                        + f"{occupancy:>6}"
                        + f"{bfactor:>6}"
                        + "           "
                        + element
                    )
                    pdb_str += line
                    pdb_str += "\n"
                    atom_index += 1
    return pdb_str


def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename, "r"):
        # skip labels
        if line[0] != ">":
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype="|S1").view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa


# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points

    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist


# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v = v / torch.norm(v, dim=-1, keepdim=True)
    w = w / torch.norm(w, dim=-1, keepdim=True)

    # this is not stable at the poles
    # vw = torch.sum(v*w, dim=-1)
    # ang = torch.acos(vw)

    # this is better
    # https://math.stackexchange.com/questions/1143354/numerically-stable-method-for-angle-between-3d-vectors/1782769
    y = torch.norm(v - w, dim=-1)
    x = torch.norm(v + w, dim=-1)
    ang = 2 * torch.atan2(y, x)

    return ang


# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1r = c - b
    b2 = d - c

    b1 = b1r / torch.norm(b1r, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)
    ang = torch.atan2(y, x)

    return ang


def outer_sum(input_tensor):
    res = input_tensor[:, None, :, :] + input_tensor[:, :, None, :]
    return res


def outer_concat(input_tensor):
    N, L, C = input_tensor.shape
    res = torch.cat(
        [
            input_tensor[:, None, :, :].repeat(1, L, 1, 1),
            input_tensor[:, :, None, :].repeat(1, 1, L, 1),
        ],
        dim=-1,
    )
    return res


def symmetry(X):
    X = 0.5 * (X + X.permute(0, 2, 1, 3))
    return X


def safenorm(X, dim=-1, epsilon=1e-10):
    return torch.sqrt(torch.square(X).sum(dim, keepdim=True) + epsilon)


def kabsch_torch(X, Y, mask):
    """Kabsch alignment of X into Y.
    Assumes X,Y are both (D, N) - usually (3, N)
    """
    #  center X and Y to the origin
    if mask.ndim == 2:
        mask = torch.unsqueeze(mask, dim=1)
    X = X * mask
    Y = Y * mask
    num_valid_pos = torch.sum(mask, dim=-1, keepdim=True)
    N, C, L = X.shape
    X_center = torch.sum(X[:, :3, :], dim=-1, keepdim=True) / num_valid_pos
    Y_center = torch.sum(Y[:, :3, :], dim=-1, keepdim=True) / num_valid_pos
    X_Ca = (X[:, :3, :] - X_center) * mask
    #    X_C  = (X[:,3:6,:] - X_center)*mask
    #    X_N = (X[:,6:,:] -X_center)*mask
    Y_Ca = (Y[:, :3, :] - Y_center) * mask
    #    Y_C = Y[:,3:6,:]
    #    Y_N = Y[:,6:,:]
    # calculate convariance matrix (for each prot in the batch)
    R = torch.matmul(X_Ca, torch.transpose(Y_Ca, 1, 2)).detach()
    # Optimal rotation matrix via SVD - warning! W must be transposed
    V, S, W = torch.svd(R)
    chi = torch.sign(torch.det(R))
    V = V.transpose(1, 2)
    W[:, :, -1] = W[:, :, -1] * chi[:, None]
    # determinant sign for direction correction
    # Create Rotation matrix U
    T = torch.matmul(W, V)
    X_Ca = torch.matmul(T, X_Ca)
    #    X_C=torch.matmul(T,X_C)
    #    X_N=torch.matmul(T,X_N)
    #    X_C=X_C-X_Ca
    #    X_N=X_N-X_Ca
    #    Y_C=Y_C/(torch.norm(Y_C,dim=1,keepdim=True)+1e-7)*1.52
    #    Y_N=Y_N/(torch.norm(Y_N,dim=1,keepdim=True)+1e-7)*1.46
    # return centered and aligned
    return X_Ca, Y_Ca


def coord2bindist(tensor, min=3.375, width=1.25, bins=15):
    dist = torch.norm(tensor.unsqueeze(1) - tensor.unsqueeze(2), dim=-1)
    dist = torch.floor((dist - min) / width).long()
    return torch.clamp(dist, 0, 14)


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(
                d_model
            )
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = torch.transpose(pe, 0, 1)
    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    return pe


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        raw = x
        x = self.norm(x)
        x = self.linear2(F.relu(self.linear1(x)))
        x = x + raw
        return x


class AxialDropout(nn.Module):
    def __init__(self, p_drop=0.1, orientation="row"):
        super(AxialDropout, self).__init__()
        assert p_drop >= 0
        self.p_drop = p_drop
        self.orient = orientation

    def forward(self, tensor):
        if not self.training or self.p_drop == 0.0:
            return tensor
        if self.orient == "row":
            broadcast_dim = 1
        else:
            broadcast_dim = 2
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - self.p_drop
        keep = torch.full(shape, keep_rate, device=tensor.device)
        keep = keep.to(tensor.dtype)
        keep = torch.bernoulli(keep)
        return keep * tensor / keep_rate


def coord2dist(coord):
    return np.linalg.norm(coord[None, :, :] - coord[:, None, :], axis=-1)


def seq2int(seq):
    # convert letters into numbers
    if type(seq) == str:
        seq = list(seq)
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype="|S1").view(np.uint8)
    msa = np.array(seq, dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa


def distmask(ca_mask):
    return ca_mask[:, None] * ca_mask[None, :]


def similarity_map(similarity):
    alphabet = np.array(list(" #=-+.|"), dtype="|S1").view(np.uint8)
    similarity = np.array(similarity, dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        similarity[similarity == alphabet[i]] = i
    return similarity


def alignment_confidence_map(confidence):
    alphabet = np.array(list("- 0123456789"), dtype="|S1").view(np.uint8)
    confidence = np.array(confidence, dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        confidence[confidence == alphabet[i]] = i
    # treat all unknown characters as gaps
    return confidence


def percent2float(x):
    return float(x.strip("%")) / 100


def costheta(Ca, N_Ca):
    L = N_Ca.shape[0]
    N_Ca = N_Ca / (np.linalg.norm(N_Ca, axis=-1, keepdims=True) + 1e-8)
    Ca_Ca = Ca[:, None, :] - Ca[None, :, :]
    Ca_Ca = Ca_Ca / (np.linalg.norm(Ca_Ca, axis=-1, keepdims=True) + 1e-8)
    N_Ca = np.tile(N_Ca[None, :, :], (L, 1, 1))
    costheta_N = np.sum(Ca_Ca * N_Ca, axis=-1)
    costheta_N = np.where(np.isnan(costheta_N), 0.0, costheta_N)
    return costheta_N


def mask_mean(mask, value):
    return torch.sum(mask * value) / (torch.sum(mask) + 1e-6)


def check_unused_pram(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)


def cross_entropy(pred, label):
    loss = -torch.sum(label * F.log_softmax(pred, dim=-1), dim=-1)
    return torch.mean(loss)


def FAPE(
    pred_frames,
    pred_atom,
    truth_frames,
    truth_atom,
    truth_frame_mask,
    truth_atom_mask,
    epsilon=1e-6,
    dmax=10,
    clamp=True,
):
    pred_frame_atom = pred_frames[:, :, None, :, None].invert_apply(
        pred_atom[:, None, :, None, :, :]
    )  # NLLT14 3
    truth_frame_atom = truth_frames[:, :, None, :, None].invert_apply(
        truth_atom[:, None, :, None, :, :]
    )
    mask = truth_frame_mask[:, :, None, :, None] * truth_atom_mask[:, None, :, None, :]
    d = mask * torch.sqrt(
        torch.square(pred_frame_atom - truth_frame_atom).sum(-1) + epsilon
    )
    if clamp:
        d = torch.clamp(d, max=dmax)
    return torch.sum(d) / (1e-6 + torch.sum(mask)) / 10.0


@torch.no_grad()
def lddt(pred, truth, mask):
    true_dist = torch.norm(truth[:, :, None, :] - truth[:, None, :, :], dim=-1)
    pred = pred.detach()
    pred_dist = torch.norm(pred[:, :, None, :] - pred[:, None, :, :], dim=-1)
    mask = mask[:, :, None] * mask[:, None, :]
    mask = mask * (true_dist < 15).float()
    mask = mask * (true_dist > 1e-5).float()
    dist_diff = torch.abs(true_dist - pred_dist)
    lddt = torch.zeros(pred.shape[:-1], dtype=pred.dtype, device=pred.device)
    for thred in [0.5, 1, 2, 4]:
        preserved = (dist_diff < thred).float()
        preserved = torch.sum(preserved * mask, dim=2)
        lddt = lddt + preserved
    lddt = lddt / (1e-6 + torch.sum(mask, dim=2)) / 4
    lddt = torch.clamp((lddt * 50).long(), min=0, max=49)
    lddt[torch.sum(mask, dim=2) < 0.1] = -100
    return lddt


def torsionAngleLoss(pred_alpha, true_alpha, alt_true_alpha, alpha_mask):
    pred_anglenorm = safenorm(pred_alpha)
    pred_alpha = pred_alpha / pred_anglenorm
    loss = torch.square(pred_alpha - true_alpha).sum(-1)
    loss = loss * alpha_mask
    alt_loss = torch.square(pred_alpha - alt_true_alpha).sum(-1)
    alt_loss = alt_loss * alpha_mask
    loss = torch.where(loss < alt_loss, loss, alt_loss)
    loss = torch.mean(loss)
    loss_anglenorm = torch.mean(
        alpha_mask
        * torch.sqrt(torch.square(torch.squeeze(pred_anglenorm, -1) - 1) + 1e-10)
    )
    loss += 0.01 * loss_anglenorm
    return loss


def check_interval(interval: Optional[List[int]] = None):
    if interval is None:
        raise ValueError("interval is None")

    if len(interval) != 2:
        raise ValueError("interval length is not 2")

    l, r = interval

    if l > r:
        raise ValueError("interval is not valid")
    if l < 0 or r < 0:
        raise ValueError("interval is not valid")


def merge_intervals(intervals: List[List[int]]):
    intervals = np.array(intervals)
    l = np.max(intervals[:, 0])
    r = np.min(intervals[:, 1])
    check_interval([l, r])
    return [l, r]


def is_subset(parent_list, son_list):
    # 将两个列表转换为集合
    parent_set = set(parent_list)
    son_set = set(son_list)

    # 检查 set2 是否是 set1 的子集
    return son_set.issubset(parent_set)


def remove_non_uppercase(s):
    return "".join(char for char in s if char.isupper())


def get_msa_one_hot(sequence):
    msa_one_hot = torch.zeros((1, 1, len(sequence), 21))
    for idx, aa in enumerate(sequence):
        msa_one_hot[:, :, idx, alpha_1.index(aa)] = 1

    return msa_one_hot
