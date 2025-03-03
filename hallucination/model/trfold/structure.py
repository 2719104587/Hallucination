import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallucination.model.alphafold2 import residue_constants
from hallucination.model.trfold.affine import Frame
from hallucination.utils import mask_mean, safenorm


class IPA(nn.Module):
    def __init__(
        self,
        Node_channels=384,
        Edge_channels=128,
        channels=192,
        groups=12,
        query_points=4,
        value_points=8,
    ):
        super(IPA, self).__init__()
        assert channels % groups == 0
        self.channels = channels
        self.groups = groups
        self.query_points = query_points
        self.value_points = value_points
        self.ppg = channels // groups
        self.query = nn.Linear(Node_channels, channels, bias=False)
        self.key = nn.Linear(Node_channels, channels, bias=False)
        self.value = nn.Linear(Node_channels, channels, bias=False)

        self.query_vector = nn.Linear(
            Node_channels, 3 * groups * query_points, bias=False
        )
        self.key_vector = nn.Linear(
            Node_channels, 3 * groups * query_points, bias=False
        )
        self.value_vector = nn.Linear(
            Node_channels, 3 * groups * value_points, bias=False
        )

        self.bias = nn.Linear(Edge_channels, groups, bias=False)
        self.gamma = nn.Parameter(torch.zeros(groups), requires_grad=True)
        self.out = nn.Linear(
            Edge_channels * groups + channels + groups * value_points * 4, Node_channels
        )

    def forward(self, Node, Edge, T):
        N, L, C = Node.shape
        query = self.query(Node).reshape(N, L, self.groups, self.ppg)
        key = self.key(Node).reshape(N, L, self.groups, self.ppg)
        value = self.value(Node).reshape(N, L, self.groups, self.ppg)

        query_v = self.query_vector(Node).reshape(
            N, L, self.groups, self.query_points, 3
        )
        key_v = self.key_vector(Node).reshape(N, L, self.groups, self.query_points, 3)
        value_v = self.value_vector(Node).reshape(
            N, L, self.groups, self.value_points, 3
        )

        T = Frame(T.rots, T.trans / 10.0)  # change unit to nano meters
        query_v = T[:, :, None, None].apply(query_v)
        query_v = query_v.unsqueeze(2)
        key_v = T[:, :, None, None].apply(key_v)
        key_v = key_v.unsqueeze(1)
        value_v = T[:, :, None, None].apply(value_v)
        value_v = value_v.float()

        square_q_subs_k = query_v - key_v
        square_q_subs_k = torch.square(square_q_subs_k)
        square_q_subs_k = square_q_subs_k.sum(dim=-1).sum(dim=-1)
        del query_v, key_v

        bias = self.bias(Edge).float()
        WC = math.sqrt(2.0 / 9 / self.query_points)
        WL = math.sqrt(1.0 / 3.0)
        query = self.query(Node).reshape(N, L, self.groups, self.ppg).float()
        key = self.key(Node).reshape(N, L, self.groups, self.ppg).float()
        value = self.value(Node).reshape(N, L, self.groups, self.ppg).float()
        attn = torch.einsum("nlgc,nmgc->nlmg", query, key / math.sqrt(self.ppg)) + bias
        del query, key, bias

        attn += -F.softplus(self.gamma) * WC / 2 * square_q_subs_k
        attn = WL * attn
        attn = F.softmax(attn, dim=2)
        attn = attn.float()
        Edge = Edge.float()

        O_Edge = torch.einsum("nijg,nijc->nigc", attn, Edge).reshape(
            N, L, self.groups * Edge.shape[-1]
        )
        O_Node = torch.einsum("nijg,njgc->nigc", attn, value).reshape(
            N, L, self.channels
        )
        O_vector = torch.einsum("nijg,njgpc->nigpc", attn, value_v)
        del attn, value_v, value
        O_vector = T[:, :, None, None].invert_apply(O_vector)
        del T
        O_vector_norm = safenorm(O_vector, dim=-1).reshape(N, L, -1)
        O_vector = O_vector.reshape(N, L, -1)

        O_Edge = O_Edge.to(Node.dtype)
        O_Node = O_Node.to(Node.dtype)
        O_vector = O_vector.to(Node.dtype)
        O_vector_norm = O_vector_norm.to(Node.dtype)
        return self.out(torch.cat([O_Edge, O_Node, O_vector, O_vector_norm], dim=-1))


class IPA_transition(nn.Module):
    def __init__(self, channels=384):
        super(IPA_transition, self).__init__()
        self.channels = channels
        self.ln = nn.LayerNorm(channels)
        self.trans = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = self.ln(x)
        x = x + self.trans(x)
        return x


class BackboneUpdate(nn.Module):
    def __init__(self, channels=384):
        super(BackboneUpdate, self).__init__()
        self.proj = nn.Linear(channels, 6)
        nn.init.constant_(self.proj.weight.data, 0)

    def forward(self, x):
        N, L, C = x.shape
        quaternion, transformation = torch.split(self.proj(x), [3, 3], dim=-1)
        b = quaternion[:, :, 0]
        c = quaternion[:, :, 1]
        d = quaternion[:, :, 2]
        a = 1.0
        R0 = a * a + b * b - c * c - d * d
        R1 = 2 * b * c - 2 * a * d
        R2 = 2 * b * d + 2 * a * c
        R3 = 2 * b * c + 2 * a * d
        R4 = a * a - b * b + c * c - d * d
        R5 = 2 * c * d - 2 * a * b
        R6 = 2 * b * d - 2 * a * c
        R7 = 2 * c * d + 2 * a * b
        R8 = a * a - b * b - c * c + d * d

        R = torch.stack([R0, R1, R2, R3, R4, R5, R6, R7, R8], dim=-1)
        R = R / (torch.square(quaternion).sum(-1, keepdim=True) + 1)
        R = R.reshape(N, L, 3, 3)
        return Frame(R, 10 * transformation)


class Torsion(nn.Module):
    def __init__(self, channels=128, in_channels=384):
        super(Torsion, self).__init__()
        self.proj_0 = nn.Linear(in_channels, channels)
        self.proj_1 = nn.Linear(in_channels, channels)

        self.block0 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

        self.block1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, 14),
        )

    def forward(self, x0, x1):
        x = self.proj_0(x0) + self.proj_1(x1)
        x = x + self.block0(x)
        x = x + self.block1(x)
        x = self.output(x)
        x = torch.reshape(x, (*x.shape[:-1], 7, 2))
        return x


class Plddt(nn.Module):
    def __init__(self, Node_channels=384):
        super(Plddt, self).__init__()
        self.ln = nn.LayerNorm(Node_channels)
        self.dense1 = nn.Linear(Node_channels, 128)
        self.dense2 = nn.Linear(128, 128)
        self.proj = nn.Linear(128, 50)

    def forward(self, x):
        x = self.ln(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)

        x = self.proj(x)
        return x


class StructureModule(nn.Module):
    def __init__(self, Node_channels=384, Edge_channels=128, blocks=8, device="cpu"):
        super(StructureModule, self).__init__()
        self.ln_s = nn.LayerNorm(Node_channels)
        self.ln_edge = nn.LayerNorm(Edge_channels)
        self.proj_s = nn.Linear(Node_channels, Node_channels)
        self.blocks = blocks
        self.IPA_layer = IPA(Node_channels=Node_channels, Edge_channels=Edge_channels)
        self.transition = IPA_transition(channels=Node_channels)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.ln_itr = nn.LayerNorm(Node_channels)
        self.S2T = BackboneUpdate(Node_channels)
        self.torsion = Torsion(128, Node_channels)
        self.plddt = Plddt()

        self.lit_frames = (
            torch.from_numpy(residue_constants.restype_rigid_group_default_frame)
            .float()
            .to(device)
        )
        self.lit_group_idx = (
            torch.from_numpy(residue_constants.restype_atom14_to_rigid_group)
            .long()
            .to(device)
        )
        self.lit_atom14_mask = (
            torch.from_numpy(residue_constants.restype_atom14_mask).float().to(device)
        )
        self.lit_positions = (
            torch.from_numpy(residue_constants.restype_atom14_rigid_group_positions)
            .float()
            .to(device)
        )

    def forward(self, Node, Edge, aatype):
        N, L, C = Node.shape
        Node_raw = self.ln_s(Node)
        Edge = self.ln_edge(Edge)
        Node = self.proj_s(Node_raw)
        R = torch.eye(3, device=Node.device)[None, None, :, :].expand(N, L, -1, -1)
        U = torch.zeros(size=(N, L, 3), device=Node.device)
        bb_frame = Frame(R, U)
        bb_traj = []
        alphas_traj = []
        for _ in range(self.blocks):
            bb_frame = bb_frame.stop_rot_gradient()
            Node = Node + self.IPA_layer(Node, Edge, bb_frame)
            Node = self.drop1(Node)
            Node = Node + self.transition(Node)
            Node = self.ln_itr(self.drop2(Node))
            bb_frame_l = self.S2T(Node)
            alphas = self.torsion(Node, Node_raw)
            bb_frame = bb_frame.compose(bb_frame_l)
            bb_traj.append(bb_frame)
            alphas_traj.append(alphas)

        frames = torsion_angles_to_frames(bb_frame, alphas, aatype, self.lit_frames)
        atoms, atom_mask = frames_and_literature_positions_to_atom14_pos(
            frames, aatype, self.lit_group_idx, self.lit_atom14_mask, self.lit_positions
        )
        plddt = self.plddt(Node)
        return bb_traj, alphas_traj, frames, atoms, atom_mask, plddt


def extreme_ca_ca_distance_violations(
    pred_atom_positions, pred_atom_mask, residue_index, max_angstrom_tolerance=1.5
):
    this_ca_pos = pred_atom_positions[:, :-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]  # (N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[:, 1:] - residue_index[:, :-1]) == 1).float()
    ca_ca_distance = torch.sqrt(
        1e-6 + torch.square(this_ca_pos - next_ca_pos).sum(dim=-1)
    )
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    violations = violations.float()
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return mask_mean(mask=mask, value=violations)


def within_residue_violations(
    atom14_pred_positions,  # (N, 14, 3)
    atom14_atom_exists,  # (N, 14)
    atom14_dists_lower_bound,  # (N, 14, 14)
    atom14_dists_upper_bound,  # (N, 14, 14)
    tighten_bounds_for_loss=0.0,
):
    dists_masks = (
        atom14_atom_exists[:, :, :, None] * atom14_atom_exists[:, :, None, :]
    ) * (1.0 - torch.eye(14, device=atom14_pred_positions.device))[None, None, :, :]
    dists = torch.sqrt(
        1e-10
        + torch.sum(
            torch.square(
                atom14_pred_positions[:, :, :, None, :]
                - atom14_pred_positions[:, :, None, :, :]
            ),
            dim=-1,
        )
    )
    dists_to_low_error = F.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = F.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    per_atom_loss_sum = torch.sum(loss, dim=-1) + torch.sum(loss, dim=-2)
    loss = torch.sum(per_atom_loss_sum) / (1e-6 + torch.sum(atom14_atom_exists))
    return loss


def between_residue_bond_loss(
    pred_atom_positions: torch.tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.tensor,  # (N, 37(14))
    residue_index: torch.tensor,  # (N)
    aatype: torch.tensor,  # (N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
):
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:, :-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]  # (N - 1)
    this_c_pos = pred_atom_positions[:, :-1, 2, :]  # (N - 1, 3)
    this_c_mask = pred_atom_mask[:, :-1, 2]  # (N - 1)
    next_n_pos = pred_atom_positions[:, 1:, 0, :]  # (N - 1, 3)
    next_n_mask = pred_atom_mask[:, 1:, 0]  # (N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[:, 1:] - residue_index[:, :-1]) == 1.0).float()
    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        1e-6 + torch.sum(torch.square(this_c_pos - next_n_pos), dim=-1)
    )
    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = (aatype[:, 1:] == residue_constants.resname_to_idx["PRO"]).float()
    gt_length = (1.0 - next_is_proline) * residue_constants.between_res_bond_length_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_c_n[1]
    gt_stddev = (
        1.0 - next_is_proline
    ) * residue_constants.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(1e-6 + torch.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = F.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    #   c_n_violation_mask = mask * (
    #       c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))
    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        1e-6 + torch.sum(torch.square(this_ca_pos - this_c_pos), axis=-1)
    )
    n_ca_bond_length = torch.sqrt(
        1e-6 + torch.sum(torch.square(next_n_pos - next_ca_pos), axis=-1)
    )
    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length.unsqueeze(-1)
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length.unsqueeze(-1)
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length.unsqueeze(-1)
    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(ca_c_n_cos_angle - gt_angle)
    )
    ca_c_n_loss_per_residue = F.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    #   ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
    #                                  (tolerance_factor_hard * gt_stddev))
    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = F.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue) / (torch.sum(mask) + 1e-6)
    #   c_n_ca_violation_mask = mask * (
    #       c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    #  per_residue_loss_sum = (c_n_loss_per_residue +
    #                          ca_c_n_loss_per_residue +
    #                          c_n_ca_loss_per_residue)
    #  per_residue_loss_sum = 0.5 * (jnp.pad(per_residue_loss_sum, [[0, 1]]) +
    #                                jnp.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    #  violation_mask = jnp.max(
    #      jnp.stack([c_n_violation_mask,
    #                 ca_c_n_violation_mask,
    #                 c_n_ca_violation_mask]), axis=0)
    #  violation_mask = jnp.maximum(
    #      jnp.pad(violation_mask, [[0, 1]]),
    #      jnp.pad(violation_mask, [[1, 0]]))
    return c_n_loss + ca_c_n_loss + c_n_ca_loss  # shape ()


def between_residue_clash_loss(
    atom14_pred_positions: torch.tensor,  # (N, 14, 3)
    atom14_atom_exists: torch.tensor,  # (N, 14)
    atom14_atom_radius: torch.tensor,  # (N, 14)
    residue_index: torch.tensor,  # (N)
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
):
    N, L, _ = atom14_atom_exists.shape
    num_atoms = 14
    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        1e-10
        + torch.sum(
            torch.square(
                atom14_pred_positions[:, :, None, :, None, :]
                - atom14_pred_positions[:, None, :, None, :, :]
            ),
            axis=-1,
        )
    )
    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[:, :, None, :, None]
        * atom14_atom_exists[:, None, :, None, :]
    )
    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask *= (
        residue_index[:, :, None, None, None] < residue_index[:, None, :, None, None]
    ).float()
    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = (
        F.one_hot(torch.tensor(2), num_classes=num_atoms)
        .float()
        .to(atom14_pred_positions.device)
    )
    n_one_hot = (
        F.one_hot(torch.tensor(0), num_classes=num_atoms)
        .float()
        .to(atom14_pred_positions.device)
    )
    neighbour_mask = (
        (residue_index[:, :, None, None, None] + 1)
        == residue_index[:, None, :, None, None]
    ).float()
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[None, None, None, :, None]
        * n_one_hot[None, None, None, None, :]
    )
    dists_mask *= 1.0 - c_n_bonds
    #   Disulfide bridge between two cysteines is no clash.
    cys_sg_idx = torch.tensor(
        residue_constants.restype_name_to_atom14_names["CYS"].index("SG")
    )
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14).float().to(dists_mask.device)

    disulfide_bonds = (
        cys_sg_one_hot[None, None, None, :, None]
        * cys_sg_one_hot[None, None, None, None, :]
    )
    dists_mask *= 1.0 - disulfide_bonds
    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[:, :, None, :, None]
        + atom14_atom_radius[:, None, :, None, :]
    )
    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * F.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )
    per_atom_loss_sum = torch.sum(dists_to_low_error, axis=[1, 3]) + torch.sum(
        dists_to_low_error, axis=[2, 4]
    )
    loss = torch.sum(per_atom_loss_sum) / (1e-6 + torch.sum(atom14_atom_exists))
    return loss


def atom_pos_valid_loss(atom14_pred_positions, atom14_atom_exists, idx, aatype):
    device = aatype.device
    dist_lower_bound = (
        torch.from_numpy(residue_constants.res_dist_bound["lower_bound"])
        .float()
        .to(device)[aatype, ...]
    )
    dist_upper_bound = (
        torch.from_numpy(residue_constants.res_dist_bound["upper_bound"])
        .float()
        .to(device)[aatype, ...]
    )
    atom14_radius = (
        torch.from_numpy(residue_constants.atom14_radius)
        .float()
        .to(device)[aatype, ...]
    )
    loss = (
        between_residue_bond_loss(
            atom14_pred_positions, atom14_atom_exists, idx, aatype
        )
        + between_residue_clash_loss(
            atom14_pred_positions, atom14_atom_exists, atom14_radius, idx
        )
        + within_residue_violations(
            atom14_pred_positions,
            atom14_atom_exists,
            dist_lower_bound,
            dist_upper_bound,
        )
    )
    return loss


@torch.no_grad()
def find_optimal_renaming(res, batch):
    #  pred_coord,gt_frame,alt_gt_frame,gt_coord,alt_gt_coord,atom_is_ambiguous,gt_coord_exists):
    pred_dists = torch.sqrt(
        1e-10
        + torch.square(
            res["atoms"][:, :, None, :, None, :] - res["atoms"][:, None, :, None, :, :]
        ).sum(-1)
    )
    gt_dists = torch.sqrt(
        1e-10
        + torch.square(
            batch["truth_atoms"][:, :, None, :, None, :]
            - batch["truth_atoms"][:, None, :, None, :, :]
        ).sum(-1)
    )
    alt_gt_dists = torch.sqrt(
        1e-10
        + torch.square(
            batch["alt_truth_atoms"][:, :, None, :, None, :]
            - batch["alt_truth_atoms"][:, None, :, None, :, :]
        ).sum(-1)
    )
    lddt = torch.abs(pred_dists - gt_dists)
    alt_lddt = torch.abs(pred_dists - alt_gt_dists)
    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (N ,N, 14, 14)
    mask = (
        batch["truth_atom_mask"][:, :, None, :, None]
        * batch["atom_is_ambiguous"][:, :, None, :, None]  # rows
        * batch["truth_atom_mask"][:, None, :, None, :]  # rows
        * (1.0 - batch["atom_is_ambiguous"][:, None, :, None, :])  # cols
    )  # cols
    alt_mask = (
        batch["alt_truth_atom_mask"][:, :, None, :, None]
        * batch["atom_is_ambiguous"][:, :, None, :, None]  # rows
        * batch["alt_truth_atom_mask"][:, None, :, None, :]  # rows
        * (1.0 - batch["atom_is_ambiguous"][:, None, :, None, :])  # cols
    )  # cols
    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (N)
    per_res_lddt = torch.sum(mask * lddt, axis=[2, 3, 4])
    alt_per_res_lddt = torch.sum(alt_mask * alt_lddt, axis=[2, 3, 4])
    alt_naming_is_better = alt_per_res_lddt < per_res_lddt
    batch["truth_atoms"] = torch.where(
        alt_naming_is_better[:, :, None, None],
        batch["alt_truth_atoms"],
        batch["truth_atoms"],
    )
    batch["truth_atom_mask"] = torch.where(
        alt_naming_is_better[:, :, None],
        batch["alt_truth_atom_mask"],
        batch["truth_atom_mask"],
    )
    batch["truth_frames"] = torch.where(
        alt_naming_is_better[:, :, None, None],
        batch["alt_truth_frames"],
        batch["truth_frames"],
    )


#  return gt_coord,gt_frame
def torsion_angles_to_frames(
    bb_frames: Frame,
    torsion: torch.Tensor,
    aatype: torch.Tensor,
    litframes: torch.Tensor,
):
    # [*, N, 8, 4, 4]
    lit_4x4 = litframes[aatype, ...]
    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_frames = Frame.from_4x4(lit_4x4)
    bb_rot = torsion.new_zeros((*((1,) * len(torsion.shape[:-1])), 2))
    bb_rot[..., 1] = 1
    # [*, N, 8, 2]
    torsion = torch.cat([bb_rot.expand(*torsion.shape[:-2], -1, -1), torsion], dim=-2)
    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    all_rots = torsion.new_zeros(default_frames.rots.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = torsion[..., 1]
    all_rots[..., 1, 2] = -torsion[..., 0]
    all_rots[..., 2, 1:] = torsion

    all_rots = Frame(all_rots, None)
    all_frames = default_frames.compose(all_rots)
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]
    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)
    all_frames_to_bb = Frame.concat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )
    all_frames_to_global = bb_frames[..., None].compose(all_frames_to_bb)
    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    all_frames: Frame,
    aatype: torch.Tensor,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14]
    group_mask = group_idx[aatype, ...]
    # [*, N, 14, 8]
    group_mask = F.one_hot(
        group_mask,
        num_classes=8,
    ).float()
    # [*, N, 14, 8]
    res_atoms_to_global = all_frames[..., None, :] * group_mask
    # [*, N, 14]
    res_atoms_to_global = res_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )
    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)
    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = res_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask
    return pred_positions, atom_mask.squeeze(-1)
