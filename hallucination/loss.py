import numpy as np
import torch
import torch.nn.functional as F
from yacs.config import CfgNode

from hallucination.model.alphafold2.residue_constants import PDB_CHAIN_IDS
from hallucination.utils import aa_1_N, get_ang, get_dih


def affine(super_xyz, ref_xyz, attach_xyz):
    super_xyz = torch.tensor(super_xyz).float()
    ref_xyz = torch.tensor(ref_xyz).float()
    attach_xyz = torch.tensor(attach_xyz).float()
    L = super_xyz.shape[0]  # xyz shape L,3,3

    super_xyz = super_xyz.reshape(L * 3, 3)
    ref_xyz = ref_xyz.reshape(L * 3, 3)
    N = attach_xyz.shape[0]  # xyz shape N,3,3

    if len(attach_xyz.shape) == 3:
        attach_xyz_atomdim = attach_xyz.shape[1]
        attach_xyz = attach_xyz.reshape(N * attach_xyz_atomdim, 3)
    else:
        attach_xyz_atomdim = None

    super_mean = torch.mean(super_xyz, dim=0, keepdim=True)
    ref_mean = torch.mean(ref_xyz, dim=0, keepdim=True)

    super_xyz = super_xyz - super_mean
    ref_xyz = ref_xyz - ref_mean

    R = torch.matmul(super_xyz.permute(1, 0), ref_xyz)
    V, S, W = torch.svd(R)
    chi = torch.ones([3, 3], device=super_xyz.device)
    chi[:, -1] = torch.sign(torch.det(V) * torch.det(W)).unsqueeze(-1)
    V = V.permute(1, 0)
    W = W * chi
    T = torch.matmul(W, V)

    attach_xyz = attach_xyz - super_mean
    attach_xyz = torch.matmul(T, attach_xyz.permute(1, 0)).permute(1, 0) + ref_mean
    if attach_xyz_atomdim:
        attach_xyz = attach_xyz.reshape(N, 3, 3)

    return attach_xyz


def calc_rog(pred_xyz, thresh):
    ca_xyz = pred_xyz[:, :, 1]
    sq_dist = torch.pow(ca_xyz - ca_xyz.mean(dim=1, keepdim=True), 2).sum(-1).mean(-1)
    rog = sq_dist.sqrt()
    return F.elu(rog - thresh) + 1


def calc_dist_rmsd(pred_xyz, ref_xyz, log=False, eps=1e-6):
    N, L = pred_xyz.shape[:2]
    pred_xyz = pred_xyz.reshape(N, L * 3, 3)
    ref_xyz = ref_xyz.reshape(1, L * 3, 3)
    D_pred_xyz = torch.triu(
        torch.cdist(pred_xyz, pred_xyz), diagonal=1
    )  # (B, L*3, L*3)
    D_ref_xyz = torch.triu(torch.cdist(ref_xyz, ref_xyz), diagonal=1)  # (1, L*3, L*3)
    n_pair = 0.5 * (3 * L * (3 * L - 1))
    diff_sq = torch.square(D_pred_xyz - D_ref_xyz).sum(dim=(-2, -1)) / n_pair
    rms_raw = torch.sqrt(diff_sq + eps)
    if log:
        rms = torch.log(rms_raw + 1.0)
    else:
        rms = rms_raw
    return rms


def calc_back_bone_fape(pred_xyz, ref_xyz, mappings=None, A=20.0, gamma=0.95):
    """
    Calculate frame-aligned point error used by Deepmind
    Input:
        - pred_xyz: predicted coordinates (B, L, n_atom, 3)
        - ref_xyz: ref_xyz coordinates (B, L, n_atom, 3)
    Output: str loss
    """

    def get_t(N, Ca, C, eps=1e-5):
        # N, Ca, C - [B, L, 3]
        # R - [B, L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
        # t - [B, L, L, 3] is the global rotation and translation invariant displacement
        v1 = N - Ca  # (B, L, 3)
        v2 = C - Ca  # (B, L, 3)
        e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)  # (B, L, 3)
        u2 = v2 - (torch.einsum("blj,blj->bl", v2, e1).unsqueeze(-1) * e1)  # (B,L,3)
        e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)  # (B, L, 3)
        e3 = torch.cross(e1, e2, dim=-1)  # (B, L, 3)
        R = torch.stack((e1, e2, e3), dim=-2)  # [B,L,3,3] - rotation matrix
        t = Ca.unsqueeze(-2) - Ca.unsqueeze(-3)  # (B,L,L,3)
        t = torch.einsum("bljk, blmk -> blmj", R, t)  # (B,L,L,3)
        return t

    t_tilde_ij = get_t(ref_xyz[:, :, 0], ref_xyz[:, :, 1], ref_xyz[:, :, 2])
    t_ij = get_t(pred_xyz[:, :, 0], pred_xyz[:, :, 1], pred_xyz[:, :, 2])
    difference = torch.norm(t_tilde_ij - t_ij, dim=-1)  # (B, L, L)
    loss = -(torch.nn.functional.relu(1.0 - difference / A)).mean(dim=(1, 2))
    return loss


def cross_entropy(pred, label, mask):
    loss = -torch.sum(label * F.log_softmax(pred, dim=-1), dim=-1) * mask
    return torch.sum(loss) / (1e-8 + torch.sum(mask))


def get_entropy_loss(distogram, mask=None, beta=10, dist_bins=36, eps=1e-16):
    def entropy(p, mask):
        S_ij = -(p * torch.log(p + eps)).sum(axis=-1)
        S_ave = torch.sum(mask * S_ij) / (torch.sum(mask) + eps)
        return S_ave

    # This loss function uses c6d
    p_dist = torch.softmax(beta * distogram[:, :, :, :dist_bins], dim=-1)
    # Mask includes all ij pairs except the diagonal by default
    if mask is None:
        L_ptn = p_dist.shape[1]
        mask = torch.ones((L_ptn, L_ptn)).to(p_dist.device)
    # Exclude diag
    L_mask = mask.shape[0]
    mask *= 1 - torch.eye(L_mask, dtype=torch.float32, device=mask.device)
    # Add batch
    mask = mask[None]
    # Modulate sharpness of probability distribution
    # Also exclude >20A bin, which improves output quality
    #    pd = torch.softmax(torch.log(beta * dict_pred['p_dist'][...,:dist_bins] + eps), axis = -1)
    # Entropy loss
    S_d = entropy(p_dist, mask)
    return S_d


def clash_loss(A_xyz, B_xyz, mask, threshold=2.0):
    dist_matrix = torch.sqrt(
        torch.square(A_xyz[:, :, None, :] - B_xyz[:, None, :, :]).sum(-1) + 1e-6
    )
    N, L = mask.shape
    dist_matrix = torch.where(
        mask.view(N, L, 1).expand_as(dist_matrix).bool(), dist_matrix, dist_matrix + 1e5
    )
    clash_loss = F.relu_(threshold - dist_matrix).sum() / (mask.sum() + 1e-6)
    return clash_loss


def xyz_to_c6d(xyz, params):
    """convert cartesian coordinates into 2d distance
    and orientation maps

    Parameters
    ----------
    xyz : pytorch tensor of shape [batch,3,nres,3]
          stores Cartesian coordinates of backbone N,Ca,C atoms
    Returns
    -------
    c6d : pytorch tensor of shape [batch,nres,nres,4]
          stores stacked dist,omega,theta,phi 2D maps
    """

    batch = xyz.shape[0]
    nres = xyz.shape[2]

    # three anchor atoms
    N = xyz[:, 0]
    Ca = xyz[:, 1]
    C = xyz[:, 2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch, nres, nres, 4], dtype=xyz.dtype, device=xyz.device)

    dist = torch.cdist(Cb, Cb, p=2)
    dist[torch.isnan(dist)] = 999.9
    c6d[..., 0] = dist + 999.9 * torch.eye(nres, device=xyz.device)[None, ...]
    b, i, j = torch.where(c6d[..., 0] < params["DMAX"])

    c6d[b, i, j, torch.full_like(b, 1)] = get_dih(
        Ca[b, i], Cb[b, i], Cb[b, j], Ca[b, j]
    )
    c6d[b, i, j, torch.full_like(b, 2)] = get_dih(N[b, i], Ca[b, i], Cb[b, i], Cb[b, j])
    c6d[b, i, j, torch.full_like(b, 3)] = get_ang(Ca[b, i], Cb[b, i], Cb[b, j])

    # fix long-range distances
    c6d[..., 0][c6d[..., 0] >= params["DMAX"]] = 999.9

    return c6d


def n_neighbors(xyz, n=1, m=9, a=0.5, b=2):
    """Gets the number of neighboring residues within a cone from each CA-CB vector.
    Inspired by formula from LayerSelector in RosettaScripts.

    Parameters
    ----------
        xyz : xyz coordinates of backbone atoms (batch, residues, atoms, xyz)
        n :   distance exponent
        m :   distance falloff midpoint
        a :   offset that controls the angular width of cone
        b :   angular sharpness exponent

    Returns
    -------
        n_nbr : number of neighbors (real-valued) for each position
    """

    c6d = xyz_to_c6d(xyz.permute(0, 2, 1, 3), {"DMAX": 20.0})

    dist = c6d[..., 0]
    phi = c6d[..., 3]
    phi[dist > 20] = np.nan
    dist[dist > 20] = np.nan

    f_dist = 1 / (1 + torch.exp(n * (dist - m)))
    f_ang = ((torch.cos(np.pi - phi) + a) / (1 + a)) ** b
    n_nbr = torch.nansum(f_dist * f_ang, axis=2)

    return n_nbr


# net charge
def nc_loss(net_out, target_charge=-7):
    i_pos = [aa_1_N[a] for a in "KR"]
    i_neg = [aa_1_N[a] for a in "ED"]
    charge = net_out["msa_one_hot"][:, 0, :][..., i_pos].sum(-1) - net_out[
        "msa_one_hot"
    ][:, 0, :][..., i_neg].sum(-1)
    loss = torch.nn.functional.relu(charge.sum(-1) - target_charge)
    return loss


# surface nonpolar
def surfnp_loss(net_out, nonpolar="VILMWF", nbr_thresh=2.5):
    i_nonpolar = [aa_1_N[a] for a in nonpolar]
    surface = 1 - torch.sigmoid(net_out["n_nbrs"] - nbr_thresh)
    surf_nonpol = net_out["msa_one_hot"][:, 0, :][..., i_nonpolar].sum(-1) * surface
    loss = surf_nonpol.sum(-1) / surface.sum(-1)
    return loss


# motif in surface or not
def motif_surf_loss(net_out, is_motif, nbr_thresh=2.5):
    surface = 1 - torch.sigmoid(net_out["n_nbrs"] - nbr_thresh)
    motif_surface = surface[:, is_motif]
    loss = motif_surface.sum(-1) / surface.sum(-1)
    factor = surface.shape[1] / is_motif.sum()
    return loss * factor


def get_coord_ca(pred_xyz, n=3):
    _, seq_length, _, _ = pred_xyz.shape
    protomer_length = seq_length // n

    coordinates = []
    for i in range(n):
        chain_id = PDB_CHAIN_IDS[i]
        for residue_index in range(i * protomer_length, (i + 1) * protomer_length):
            coordinates.append(
                [
                    chain_id,
                    residue_index,
                    pred_xyz[:, residue_index, 1].cpu().detach().numpy(),
                ]
            )
    return np.array(coordinates, dtype=object)


def lj_rep(d, sigma, epsilon=1):
    """
    Input:
        - d: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    """
    # Alford et al., JCTC 2017
    m = -12 * epsilon / (0.6 * sigma) * ((1 / 0.6) ** 12 + (1 / 0.6) ** 6)
    E0 = epsilon * ((1 / 0.6) ** 12 - 2 * (1 / 0.6) ** 6 + 1)
    E_near = m * d + (E0 - m * 0.6 * sigma)
    E_mid = epsilon * ((sigma / d) ** 12 - 2 * (sigma / d) ** 6 + 1)
    return (d <= 0.6 * sigma) * E_near + ((d > 0.6 * sigma) & (d <= sigma)) * E_mid


def calc_lj_rep(lig_dist, sigma, epsilon=1):
    """
    Calculate repulsion loss

    Input:
        - lig_dist: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    Output:
        - loss:     Lennard-Jones-like repulsive loss
    """
    loss = lj_rep(lig_dist, sigma=sigma, epsilon=epsilon)
    loss[torch.isnan(loss)] = 0

    # return torch.mean(loss, dim=[1,2])
    return torch.sum(loss, dim=[1, 2]) / 1e5


def calc_lj_atr(lig_dist, sigma, epsilon=1):
    """
    Calculate attraction loss

    Input:
        - lig_dist: distances from motif bb atoms to ligand atoms
        - sigma:    inter-atomic distance for repulsion
        - epsilon:  scale term for repulsion
    Output:
        - loss:     Lennard-Jones-like attractive loss
    """

    def lj_atr(d, sigma, epsilon=1):
        # Alford et al., JCTC 2017
        E_mid = epsilon * ((sigma / d) ** 12 - 2 * (sigma / d) ** 6)
        return (d <= sigma) * (-epsilon) + (d > sigma) * E_mid

    loss = lj_atr(lig_dist, sigma=sigma, epsilon=epsilon)
    loss[torch.isnan(loss)] = 0

    # return torch.mean(loss, dim=[1,2])
    return torch.sum(loss, dim=[1, 2]) / 1e5


def set_dist_to_target(pred_xyz, coord, paste_loc):
    B = pred_xyz.shape[0]
    target_ccan, contig_ccan = coord[0], coord[1]
    motif_ccan = (
        pred_xyz[:, paste_loc[0] : paste_loc[1], :3, :].squeeze().detach().cpu().numpy()
    )
    target_ccan_sup = torch.tensor(affine(contig_ccan, motif_ccan, target_ccan)).float()
    target_ccan_sup = (
        torch.unsqueeze(target_ccan_sup, dim=0).reshape(B, -1, 3).to(pred_xyz.device)
    )
    pred_ccan = pred_xyz[:, :, :3, :].reshape(B, -1, 3)

    dist = (
        (pred_ccan[:, :, None, :] - target_ccan_sup[:, None, :, :])
        .pow(2)
        .sum(-1)
        .sqrt()
    )  # (B, n_hal, n_rep)
    return dist


def hl_score(
    net_out,
    motif_truth_list,
    is_motif_list,
    motif_aatype_list,
    motif_target,
    target_coord,
    paste_locs_list,
    cfg: CfgNode,
    clean_code,
    strategy,
    is_gd=False,
    predict_model="trfold",
):
    loss = {"tot": 0}
    tot_loss = 0
    pred_xyz = net_out["coord"]

    clean_code_part = strategy[2]
    is_symmetry = strategy[1]
    is_root_node = len(clean_code_part) == len(clean_code)
    index_section = strategy[3]
    if len(clean_code_part) == 1:
        assert len(index_section) == 1
        rog = calc_rog(
            pred_xyz[:, index_section[0][0] : index_section[0][1], :, :],
            cfg.loss.rog_thresh,
        )[0]
        loss["rog"] = cfg.loss.w_rog * rog
        tot_loss += loss["rog"]

    elif len(clean_code_part) > 1 and is_symmetry:
        tot_rog = 0
        for section in index_section:
            tot_rog += calc_rog(
                pred_xyz[:, section[0] : section[1], :, :], cfg.loss.rog_thresh
            )[0]
        loss["rog"] = cfg.loss.w_rog * tot_rog / len(index_section)
        tot_loss += loss["rog"]

    if cfg.loss.w_nc != 0 and is_root_node and len(motif_truth_list) > 0:
        nc = nc_loss(net_out, cfg.loss.charge_thresh)[0]
        loss["nc"] = cfg.loss.w_nc * nc
        tot_loss += loss["nc"]

    if cfg.loss.w_surfnp != 0 and is_root_node and len(motif_truth_list) > 0:
        net_out["n_nbrs"] = n_neighbors(net_out["coord"])
        surfnp = surfnp_loss(net_out)[0]
        loss["surfnp"] = cfg.loss.w_surfnp * surfnp
        tot_loss += loss["surfnp"]

    if cfg.loss.w_ptm != 0 and is_root_node:
        if predict_model == "trfold":
            e_probs = torch.softmax(net_out["ptm"], dim=-1)
            n_res = e_probs.shape[1]
            d0 = 1.24 * np.cbrt(max(n_res, 19) - 15) - 1.8
            e_bin_centers = 0.25 + torch.arange(64).float().cuda() * 0.5
            score = 1.0 / (1 + (e_bin_centers / d0) ** 2)
            ptm = torch.max(torch.sum(e_probs * score, dim=-1).mean(-1))
            loss["ptm"] = -ptm * cfg.loss.w_ptm
        elif predict_model == "alphafold":
            loss["ptm"] = cfg.loss.w_ptm * (1 - torch.mean(net_out["ptm"]))
        tot_loss += loss["ptm"]

    if cfg.loss.w_ptm != 0 and is_root_node and predict_model == "alphafold":
        loss["pae"] = cfg.loss.w_pae * np.mean(net_out["pae"])
        tot_loss += loss["pae"]

    distogram = net_out["distogram"]
    if cfg.loss.w_entropy != 0 and is_root_node:
        dist_entropy = get_entropy_loss(distogram)
        loss["entropy"] = cfg.loss.w_entropy * dist_entropy
        tot_loss += loss["entropy"]

    if len(motif_truth_list) > 0 and is_root_node:
        if cfg.loss.w_rep != 0 or cfg.loss.w_atr != 0 and motif_target is not None:
            rep_loss_list = []
            atr_loss_list = []
            for (
                coord,
                paste_loc,
            ) in zip(target_coord, paste_locs_list):
                dist = set_dist_to_target(pred_xyz, coord, paste_loc)

                if cfg.loss.w_rep != 0:
                    rep_loss = calc_lj_rep(dist, cfg.loss.rep_sigma)
                    rep_loss_list.append(rep_loss)

                if cfg.loss.w_atr != 0:
                    atr_loss = calc_lj_atr(dist, cfg.loss.atr_sigma)
                    atr_loss_list.append(atr_loss)

            if cfg.loss.w_rep != 0:
                loss["rep_loss"] = cfg.loss.w_rep * torch.mean(
                    torch.tensor(rep_loss_list)
                )
                tot_loss += loss["rep_loss"]

            if cfg.loss.w_atr != 0:
                loss["atr_loss"] = cfg.loss.w_atr * torch.mean(
                    torch.tensor(atr_loss_list)
                )
                tot_loss += loss["atr_loss"]

        motif_surf_list = []
        dist_cce_list = []
        fape_list = []
        rmsd_list = []
        for is_motif, motif_truth, motif_aatype in zip(
            is_motif_list, motif_truth_list, motif_aatype_list
        ):
            if cfg.loss.w_motif_surf != 0 and not is_gd:
                net_out["n_nbrs"] = n_neighbors(net_out["coord"])
                motif_surf = motif_surf_loss(net_out, is_motif)[0]
                motif_surf_list.append(motif_surf)

            if cfg.loss.w_cce != 0:
                is_GLY = (motif_aatype == 7).float()
                motif_GCB = (
                    motif_truth[:, :, 4, :] * (1 - is_GLY[:, :, None])
                    + motif_truth[:, :, 1, :] * is_GLY[:, :, None]
                )
                motif_distogram = torch.norm(
                    motif_GCB[:, :, None, :] - motif_GCB[:, None, :, :], dim=-1
                )
                dist_mask = (motif_distogram < 20).float() * (
                    motif_distogram > 1e-4
                ).float()
                motif_distogram = F.one_hot(
                    torch.clip(((motif_distogram - 2) / 0.3125).long(), min=0, max=63),
                    num_classes=64,
                ).float()
                distogram_temp = distogram[:, is_motif, :, :]
                distogram_temp = distogram_temp[:, :, is_motif, :]
                dist_cce = cross_entropy(distogram_temp, motif_distogram, dist_mask)
                dist_cce_list.append(dist_cce.item())

            pred_motif_xyz = pred_xyz[:, is_motif, :, :]
            if cfg.loss.w_fape != 0:
                fape = calc_back_bone_fape(pred_motif_xyz, motif_truth)
                fape_list.append(fape.squeeze(0).item())

            motif_truth = motif_truth[:, :, :3, :].squeeze()
            pred_motif_xyz = pred_motif_xyz[:, :, :3, :].squeeze()

            if cfg.loss.w_rmsd != 0:
                motif_truth_sup = affine(motif_truth, pred_motif_xyz, motif_truth)
                motif_truth = torch.unsqueeze(motif_truth_sup, dim=0)
                rmsd = torch.sqrt(
                    torch.mean(torch.square(pred_motif_xyz - motif_truth).sum(-1))
                    + 1e-6
                )
                rmsd = torch.clip(rmsd, min=cfg.loss.rmsd_thresh)
                rmsd_list.append(rmsd.item())

        if cfg.loss.w_motif_surf != 0 and not is_gd:
            loss["motif_surf"] = (
                cfg.loss.w_motif_surf * torch.mean(torch.tensor(motif_surf_list)) * -1
            )
            tot_loss += loss["motif_surf"]

        if cfg.loss.w_cce != 0:
            loss["dist_cce"] = cfg.loss.w_cce * torch.mean(torch.tensor(dist_cce_list))
            tot_loss += loss["dist_cce"]

        if cfg.loss.w_fape != 0:
            loss["fape"] = cfg.loss.w_fape * torch.mean(torch.tensor(fape_list))
            tot_loss += loss["fape"]

        if cfg.loss.w_rmsd != 0:
            loss["rmsd"] = cfg.loss.w_rmsd * torch.mean(torch.tensor(rmsd_list))
            tot_loss += loss["rmsd"]

    if cfg.loss.w_plddt != 0 and is_root_node:
        if predict_model == "trfold":
            bin_values = torch.arange(50).float().cuda() * 0.02 + 0.01
            plddt = torch.sum(
                torch.softmax(net_out["plddt"], dim=-1) * bin_values, dim=-1
            )
            plddt = torch.mean(F.relu(0.9 - plddt))
            loss["plddt"] = cfg.loss.w_plddt * plddt
        elif predict_model == "alphafold":
            loss["plddt"] = cfg.loss.w_plddt * (1 - torch.mean(net_out["plddt"]))
        tot_loss += loss["plddt"]

    ## plane symmetry
    if cfg.loss.w_symmetry != 0 and is_symmetry:
        symmetry_index_section = strategy[3]
        assert symmetry_index_section is not None
        symmetry_part = []
        n_order = len(symmetry_index_section)
        for start, end in symmetry_index_section:
            symmetry_part.append(pred_xyz[:, start:end, :, :])
        symmetry_coord = torch.cat(symmetry_part, dim=1)
        c = get_coord_ca(symmetry_coord, n_order)  # get CA atoms
        # Compute center of mass (CA) of each chain.
        chains = set(c[:, 0])
        center_of_mass = {ch: float for ch in chains}
        for ch in chains:
            center_of_mass[ch] = np.mean(c[c[:, 0] == ch][:, 2:], axis=0)[0]

        if cfg.loss.symmetry_type == "plane":
            # Compare distances between adjacent chains, including first-last.
            chain_order = sorted(center_of_mass.keys())
            separation_std = 0
            for i in range(n_order // 2):
                next_chain = np.roll(chain_order, i + 1)
                proto_dist = []
                for k, ch in enumerate(chain_order):
                    proto_dist.append(
                        np.linalg.norm(
                            center_of_mass[next_chain[k]] - center_of_mass[ch]
                        )
                    )  # compute separation distances.
                separation_std += np.std(proto_dist)
            loss["symmetry"] = cfg.loss.w_symmetry / (n_order // 2) * separation_std

        elif cfg.loss.symmetry_type == "space":
            center_of_mass_oligomer = np.mean(c[:, 2:], axis=0)[0]

            all_proto_dist = []
            chain_center_dist = []
            for ch1 in chains:
                proto_dist = []
                chain_center_dist.append(
                    np.linalg.norm(center_of_mass[ch1] - center_of_mass_oligomer)
                )
                for ch2 in chains:
                    proto_dist.append(
                        np.linalg.norm(center_of_mass[ch1] - center_of_mass[ch2])
                    )
                all_proto_dist.append(sorted(proto_dist))
            all_proto_dist_array = np.array(all_proto_dist)

            separation_std = np.std(chain_center_dist)
            for i in range(all_proto_dist_array.shape[1]):
                separation_std += np.std(all_proto_dist_array[:, i])

            loss["symmetry"] = (
                cfg.loss.w_symmetry
                * separation_std
                / (all_proto_dist_array.shape[1] + 1)
            )
        else:
            loss["symmetry"] = 0

        tot_loss += loss["symmetry"]

    # ## space symmetry
    # if cfg.loss.w_symmetry > 0 and is_symmetry:
    #     symmetry_index_section = strategy[3]
    #     assert symmetry_index_section is not None
    #     symmetry_part = []
    #     n_order = len(symmetry_index_section)
    #     for start, end in symmetry_index_section:
    #         symmetry_part.append(pred_xyz[:, start:end, :, :])
    #     symmetry_coord = torch.cat(symmetry_part, dim=1)
    #     c = get_coord_ca(symmetry_coord, n_order) # get CA atoms
    #     # Compute center of mass (CA) of each chain.
    #     chains = set(c[:,0])
    #     center_of_mass = {ch:float for ch in chains}
    #     for ch in chains:
    #         center_of_mass[ch] = np.mean(c[c[:,0]==ch][:,2:], axis=0)[0]

    #     tot_loss += loss["symmetry"]

    loss["tot"] = tot_loss

    return loss
