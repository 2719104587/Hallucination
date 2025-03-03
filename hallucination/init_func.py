import torch
import torch.nn.functional as F
import numpy as np
import json
from Bio.PDB import PDBParser
from typing import List, Dict, Optional

from hallucination.model.alphafold2.residue_constants import (
    chain_ids,
    restype_3to1,
    HHBLITS_AA_TO_ID,
)
from hallucination.utils import read_pdb

alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype="|S1").view(np.uint8)


def convert_msa_from_letter_to_int(seqs):
    msa = np.array([list(s) for s in seqs], dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    msa[msa > 20] = 20
    return msa


def convert_seq_from_letter_to_int(seq):
    msa = np.array(list(seq), dtype="|S1").view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i
    msa[msa > 20] = 20
    return msa


def to_clean(input):
    clean_str = ""
    for i in str(input):
        if i in chain_ids:
            clean_str += i
    return clean_str


def get_max_deepth(node):
    deepth_list = []
    que = [node]
    while que:
        cur_node = que.pop(0)
        deepth_list.append(cur_node.deepth)
        for subnode in cur_node.children:
            que.append(subnode)
    return max(deepth_list)


def get_seq(node):
    return node.seq


def is_allstr(node):
    for info in node.node_info():
        if type(info) != str:
            return False
    return True


class TreeNode:
    COMB = 0
    SYMM = 1

    def __init__(self, parent=None):
        self.children: List["TreeNode"] = []
        self.parent_node: "TreeNode" = parent
        self.mode = self.COMB
        self.seq: str = ""
        self.seq_index = []
        self.deepth = None

    def node_info(self):
        if self.children:
            node_list = []
            node_index = []
            for c in self.children:
                node_list.append(c.node_info())
                node_index.extend(c.seq_index)
            node_index = sorted(node_index)
            self.seq_index = node_index
            if len(set(node_list)) == 1 and len(node_list) > 1:
                self.mode = self.SYMM
            return tuple(node_list)
        if len(set(list(self.seq))) == 1 and len(self.seq) > 1:
            self.mode = self.SYMM
        return self.seq


class SeqTree:
    def __init__(self, str_in: str):
        self.seq = str_in
        self.tree_root: TreeNode = TreeNode()
        self.__parse()
        self.tree_sort()

    def __parse(self):
        temp_node: TreeNode = self.tree_root
        str_index = 0
        deepth = 0
        for c in self.seq:
            if c == "(":
                temp_node.children.append(TreeNode(parent=temp_node))
                deepth += 1
                temp_node = temp_node.children[-1]
                temp_node.deepth = deepth
            elif c == ")":
                temp_node = temp_node.parent_node
                deepth -= 1
                temp_node.deepth = deepth
            else:
                temp_node.seq += c
                temp_node.seq_index.append(str_index)
                str_index += 1

    def tree_sort(self):
        all_subnode_list = [self.tree_root]
        while all_subnode_list:
            cur_subnode = all_subnode_list.pop(0)
            temp_children = sorted(
                cur_subnode.children, key=get_max_deepth, reverse=True
            )
            str_index = 0
            for index, tempnode in enumerate(temp_children):
                if is_allstr(tempnode):
                    str_index == index
                    break
            temp_children = temp_children[:str_index] + sorted(
                temp_children[str_index:], key=get_seq
            )
            cur_subnode.children = temp_children
            for subnode in cur_subnode.children:
                all_subnode_list.append(subnode)


def get_template_feature(aatype, atoms):
    L = aatype.size()[0]
    template_seq = aatype.clone().unsqueeze(0)

    template_CA = atoms[:, :, 1, :]
    template_CB = atoms[:, :, 3, :]
    is_GLY = template_seq == 7
    template_GCB = torch.where(
        is_GLY.unsqueeze(-1).expand_as(template_CA), template_CA, template_CB
    )
    template_dist = torch.norm(
        template_GCB[:, :, None, :] - template_GCB[:, None, :, :], dim=-1
    )
    template_GCB_mask = torch.norm(template_CA, dim=-1) > 1e-5
    template_dist_mask = (
        template_GCB_mask[:, :, None] * template_GCB_mask[:, None, :]
    ).float()
    template_dist = (template_dist - 3.25) / 1.25
    template_dist = torch.clip(template_dist.long(), min=0, max=38)
    template_dist = F.one_hot(template_dist, num_classes=39).float()
    template_dist = torch.multiply(template_dist, template_dist_mask[:, :, :, None])

    template_seq = torch.where(
        template_GCB_mask, template_seq, torch.full_like(template_seq, 20)
    )
    template_seq = torch.clamp(template_seq, max=20)
    template_seq = F.one_hot(template_seq, num_classes=21).float()
    template = torch.cat(
        [
            template_dist,
            template_dist_mask.unsqueeze(-1),
            template_seq[:, None, :, :].repeat(1, L, 1, 1),
            template_seq[:, :, None, :].repeat(1, 1, L, 1),
        ],
        dim=-1,
    )

    return template


def get_motif_gaps(
    motif_gaps: Optional[List[List[int]]] = None,
    motif_gaps_cluster: Optional[Dict] = None,
):
    if motif_gaps is None:
        return None

    if motif_gaps_cluster is None:
        result_motif_gaps = []
        for l, r in motif_gaps:
            if r <= 0:
                result_motif_gaps.append(0)
            else:
                result_motif_gaps.append(np.random.randint(l, r))

        return result_motif_gaps

    motif_gaps_length = {}
    for cluster_id, motif_idx_list in motif_gaps_cluster.items():
        if motif_gaps[motif_idx_list[0]][1] <= 0:
            gap_length = 0
        else:
            gap_length = np.random.randint(
                motif_gaps[motif_idx_list[0]][0], motif_gaps[motif_idx_list[0]][1]
            )
        for idx in motif_idx_list:
            motif_gaps_length[idx] = gap_length

    return [motif_gaps_length[i] for i in range(len(motif_gaps))]


def get_motif_gaps_aa_map(motif_locs_list, motif_gaps, motif_gaps_cluster):
    if motif_locs_list is None or motif_gaps is None or motif_gaps_cluster is None:
        return None

    motif_gaps_aa_idx = {}
    length = 0
    for i in range(len(motif_gaps) - 1):
        motif_gaps_aa_idx[i] = [idx for idx in range(length, length + motif_gaps[i])]
        length += motif_gaps[i]
        length += motif_locs_list[i][1] - motif_locs_list[i][0] + 1
    motif_gaps_aa_idx[len(motif_gaps) - 1] = [
        idx for idx in range(length, length + motif_gaps[-1])
    ]

    motif_gaps_aa_idx_map = {}
    for cluster_id, motif_idx_list in motif_gaps_cluster.items():
        related_aa_list = [motif_gaps_aa_idx[idx] for idx in motif_idx_list]

        for gap_aa_index_list in related_aa_list:
            for index, gap_aa_index in enumerate(gap_aa_index_list):
                motif_gaps_aa_idx_map[gap_aa_index] = [
                    aa_list[index] for aa_list in related_aa_list
                ]

    return motif_gaps_aa_idx_map


def get_length_tuple(length_tuple, motif_locs, motif_gaps: Optional[list[int]] = None):
    if motif_gaps is None:
        return length_tuple
    else:
        length = 0
        for i in range(len(motif_gaps) - 1):
            length += motif_gaps[i]
            length += motif_locs[i][1] - motif_locs[i][0] + 1
        length += motif_gaps[-1]
        return (length,)


def get_length_dict(clean_code, length_tuple):
    length_dict = {}
    length_index = 0
    for letter in chain_ids:
        if letter in clean_code:
            length_dict[letter] = length_tuple[length_index]
            length_index += 1
    return length_dict


def get_same_code_aa_map(clean_code, length_dict):
    same_code_index = {}
    length = 0
    for unit in clean_code:
        if unit not in same_code_index.keys():
            same_code_index[unit] = []
        same_code_index[unit].append(
            [i for i in range(length, length + length_dict[unit])]
        )
        length += length_dict[unit]

    same_code_aa_map = {}
    for unit, related_aa_list in same_code_index.items():
        for aa_index_list in related_aa_list:
            for index, aa_index in enumerate(aa_index_list):
                same_code_aa_map[aa_index] = [
                    aa_list[index] for aa_list in related_aa_list
                ]

    return same_code_aa_map


def get_loss_strategy(code, clean_code, length_dict):
    loss_strategy = []

    tree = SeqTree(code)
    que = tree.tree_root.children
    while que:
        cur_node = que.pop(0)
        for i in cur_node.children:
            que.append(i)

        node_info = cur_node.node_info()

        is_symmetry = cur_node.mode
        seq_index = cur_node.seq_index

        index_section = []
        if is_symmetry:
            n_order = len(node_info)
            unit_length = len(seq_index) / n_order
            for i in range(n_order):
                start = 0
                for letter in clean_code[: seq_index[int(i * unit_length)]]:
                    start += length_dict[letter]
                end = start
                for letter in to_clean(node_info[i]):
                    end += length_dict[letter]
                index_section.append([start, end])
        else:
            start = 0
            for letter in clean_code[: seq_index[0]]:
                start += length_dict[letter]
            end = start
            for letter in to_clean(node_info):
                end += length_dict[letter]
            index_section.append([start, end])

        loss_strategy.append(
            [node_info, is_symmetry, to_clean(node_info), index_section]
        )
    return loss_strategy


def get_chain_info(code, length_dict):
    replace_code = code.replace("(", "-").replace(")", "-")
    replace_code_split = replace_code.split("-")
    chain_info = []
    for split in replace_code_split:
        if split not in ["-", ""]:
            chain_length = 0
            for chain in split:
                chain_length += length_dict[chain]
            chain_info.append([split, chain_length])
    return chain_info


def get_unit_index(clean_code, length_dict, init_method, sequence):
    start_index = 0
    unit_index_dict = {}
    for letter in clean_code:
        if letter not in unit_index_dict.keys():
            unit_index_dict[letter] = []
        unit_index_dict[letter].append([start_index, start_index + length_dict[letter]])
        start_index += length_dict[letter]
    if init_method != "random":
        for unit, index_list in unit_index_dict.items():
            unit_sequence = sequence[index_list[0][0] : index_list[0][1]]
            for start, end in index_list:
                assert sequence[start:end] == unit_sequence
    return unit_index_dict


def get_idx(code, length_dict):
    assist_idx = 0
    idx = []
    for char_idx, char in enumerate(code):
        if char in chain_ids:
            idx.extend([assist_idx + i for i in range(length_dict[char])])
            assist_idx += length_dict[char]
            if code[char_idx + 1] in ["(", ")"]:
                assist_idx += 200
    idx = torch.tensor(idx)
    return idx


def get_residue_ccan(residue):
    atoms = list(residue.get_atoms())
    x_c, y_c, z_c = None, None, None
    x_ca, y_ca, z_ca = None, None, None
    x_n, y_n, z_n = None, None, None
    for atom in atoms:
        if atom.name == "C":
            x_c = atom.get_vector()[0]
            y_c = atom.get_vector()[1]
            z_c = atom.get_vector()[2]
        elif atom.name == "CA":
            x_ca = atom.get_vector()[0]
            y_ca = atom.get_vector()[1]
            z_ca = atom.get_vector()[2]
        elif atom.name == "N":
            x_n = atom.get_vector()[0]
            y_n = atom.get_vector()[1]
            z_n = atom.get_vector()[2]
    if x_c is None or y_c is None or z_c is None:
        print("residue have not C")
        exit()
    elif x_ca is None or y_ca is None or z_ca is None:
        print("residue have not CA")
        exit()
    elif x_n is None or y_n is None or z_n is None:
        print("residue have not N")
        exit()
    return [[x_c, y_c, z_c], [x_ca, y_ca, z_ca], [x_n, y_n, z_n]]


def get_target_ccan(target_pdb, target_chains, contig_loc, contig_chain):
    parser = PDBParser(PERMISSIVE=True)
    struct = parser.get_structure("X", target_pdb)
    chains = list(struct[0].get_chains())

    target_xyz_ccan = []
    contig_xyz_ccan = []

    for chain in chains:
        chain_id = chain.get_id()
        residues = list(chain.get_residues())

        for residue in residues:
            residue_name = residue.get_resname()
            if residue_name not in restype_3to1.keys():
                continue
            residue_id = residue.id[1]
            residue_xyz_ccan = get_residue_ccan(residue)
            if chain_id in target_chains:
                target_xyz_ccan.append(residue_xyz_ccan)
            if (
                chain_id == contig_chain
                and contig_loc[0] <= residue_id <= contig_loc[1]
            ):
                contig_xyz_ccan.append(residue_xyz_ccan)
    return target_xyz_ccan, contig_xyz_ccan


def get_paste_locs(
    motif_locs,
    motif_gaps,
    length_tuple,
    paste_locs,
    chain_info,
    sum_length,
    special_motif_index=[],
    special_motif_expand=[],
):
    if len(motif_locs) <= 0:
        return []
    if motif_gaps is not None:
        paste_locs_list = []
        index = 0
        for i in range(len(motif_gaps) - 1):
            index += motif_gaps[i]
            paste_locs_list.append(
                [index, index + motif_locs[i][1] - motif_locs[i][0] + 1]
            )
            index += motif_locs[i][1] - motif_locs[i][0] + 1

        # oligomer_paste_locs_list = []
        # for i in range(3):
        #     for paste_loc in paste_locs_list:
        #         oligomer_paste_locs_list.append(
        #             [
        #                 paste_loc[0] + i * length_tuple[0],
        #                 paste_loc[1] + i * length_tuple[0],
        #             ]
        #         )
        # return oligomer_paste_locs_list

        return paste_locs_list

    if paste_locs is not None and len(paste_locs) > 0:
        paste_locs_list = paste_locs
        assert len(paste_locs_list) == len(motif_locs)
        for paste_loc, motif_loc in zip(paste_locs_list, motif_locs):
            try:
                assert paste_loc[1] - paste_loc[0] == motif_loc[1] - motif_loc[0] + 1
            except:
                print(paste_loc)
                print(motif_loc)
                assert paste_loc[1] - paste_loc[0] == motif_loc[1] - motif_loc[0] + 1
        return paste_locs_list

    else:
        nearest_rupture = {}
        chain_index = 0
        length_node = chain_info[chain_index][1]
        for i in range(sum_length):
            if i >= length_node:
                chain_index += 1
                length_node += chain_info[chain_index][1]
            assert length_node >= i
            nearest_rupture[i] = length_node

        max_times = 10000
        for i in range(max_times):
            paste_locs_list = []
            temp_paste_locs_list = []
            random_list = [j for j in range(sum_length)]

            for motif_index, motif in enumerate(motif_locs):
                # ### temp
                # if motif_index == len(motif_locs) - 1:
                #     p = np.random.choice([0, 1])
                #     if p == 1:
                #         start = 0
                #         end = motif[1] - motif[0]
                #     else:
                #         start = sum_length - 24
                #         end = start + motif[1] - motif[0]
                # # elif motif_index == len(motif_locs) -2:
                # #     p = np.random.choice([0, 1])
                # #     if p == 1:
                # #         start = 0
                # #         end = motif[1] - motif[0]
                # #     else:
                # #         start = sum_length - 25
                # #         end = start + motif[1] - motif[0]
                # else:
                #     start = random_list[np.random.randint(0, len(random_list))]
                #     end = start + motif[1] - motif[0]
                start = random_list[np.random.randint(0, len(random_list))]
                end = start + motif[1] - motif[0] + 1
                if motif_index in special_motif_index:
                    expand_f = special_motif_expand[
                        special_motif_index.index(motif_index)
                    ][0]
                    expand_b = special_motif_expand[
                        special_motif_index.index(motif_index)
                    ][1]
                    end += expand_f + expand_b

                # # temp
                # if start < 20 or end > sum_length -20:
                #     continue

                if end <= nearest_rupture[start]:
                    is_append = True
                    for locs in temp_paste_locs_list:
                        if locs[0] <= start <= locs[1]:
                            is_append = False
                        elif start <= locs[0] <= end:
                            is_append = False
                    if is_append:
                        temp_paste_locs_list.append([start, end])
                        if motif_index in special_motif_index:
                            paste_locs_list.append([start + expand_f, end - expand_b])
                        else:
                            paste_locs_list.append([start, end])
                        for k in range(start, end):
                            random_list.remove(k)
                else:
                    break
            if len(paste_locs_list) == len(motif_locs):
                return paste_locs_list
        return []


def get_force_info(
    clean_code,
    length_dict,
    force_idx,
    force_aa,
    paste_locs_list,
    special_motif_index,
    special_motif_expand,
):
    force_dict = {}
    motif_force_dict = {}
    start_index = 0
    for unit in clean_code:
        if unit not in force_dict.keys():
            force_dict[unit] = {"force_idx": [], "force_aa": []}
            motif_force_dict[unit] = []
        unit_index = [i for i in range(start_index, start_index + length_dict[unit])]

        for single_force_idx, single_force_aa in zip(force_idx, force_aa):
            if single_force_idx in unit_index:
                force_dict[unit]["force_idx"].append(unit_index.index(single_force_idx))
                force_dict[unit]["force_aa"].append(single_force_aa)

        for motif_index, paste_loc in enumerate(paste_locs_list):
            if motif_index in special_motif_index:
                expand_f = special_motif_expand[special_motif_index.index(motif_index)][
                    0
                ]
                expand_b = special_motif_expand[special_motif_index.index(motif_index)][
                    1
                ]
                start = paste_loc[0] - expand_f
                end = paste_loc[1] + expand_b
            else:
                start = paste_loc[0]
                end = paste_loc[1]
            for idx in range(start, end):
                if idx in unit_index:
                    motif_force_dict[unit].append(unit_index.index(idx))

        start_index += length_dict[unit]
    return force_dict, motif_force_dict


def get_mutate_idx_list(mutate_idx, paste_locs_list):
    if mutate_idx is None:
        return None
    elif len(mutate_idx) == 0:
        return None
    else:
        mutate_idx_list = []
        mutate_idx_split = [
            sub_str for sub_str in mutate_idx.split(",") if len(sub_str) > 0
        ]
        for mutate_loc in mutate_idx_split:
            mutate_loc_split = mutate_loc.split("-")
            if len(mutate_loc_split) == 2:
                start = int(mutate_loc_split[0])
                end = int(mutate_loc_split[1])
            else:
                start = int(mutate_loc_split[0])
                end = start
            mutate_idx_list.extend([i for i in range(start, end)])

        mutate_idx_list = list(set(mutate_idx_list))
        mutate_idx_list = sorted(mutate_idx_list)

        if len(paste_locs_list) > 0:
            paste_idx_list = []
            for paste_loc in paste_locs_list:
                paste_idx_list.extend([i for i in range(paste_loc[0], paste_loc[1])])

            insec = set(paste_idx_list).intersection(mutate_idx_list)
            if len(insec) > 0:
                return []

        return mutate_idx_list


def make_fold_params(sequence):
    msa = [sequence]
    msa = convert_msa_from_letter_to_int(msa)
    msa = torch.from_numpy(msa).long()
    aatype = msa[0, :].clone()
    return msa, aatype


def get_template_trfold(
    sum_length,
    paste_locs_list,
    read_pdb_index,
    motif_template,
    motif_locs_list,
    motif_locs_chain_list,
    motif_pdb_path,
):
    atoms_init = torch.zeros((1, sum_length, 37, 3))
    aatype_init = torch.full((sum_length,), 20)
    for paste_loc, pdb_index, template_stat, motif_loc, motif_loc_chain in zip(
        paste_locs_list,
        read_pdb_index,
        motif_template,
        motif_locs_list,
        motif_locs_chain_list,
    ):
        if template_stat:
            motif_seq, motif_template_atoms, _ = read_pdb(
                motif_pdb_path[pdb_index], motif_loc, motif_loc_chain, "trfold"
            )
            _, motif_aatype = make_fold_params(motif_seq)
            atoms_init[:, paste_loc[0] : paste_loc[1], :, :] = motif_template_atoms
            aatype_init[paste_loc[0] : paste_loc[1]] = motif_aatype
    template = get_template_feature(
        aatype_init, atoms_init[:, :, :5, :]
    )  # indel change
    return template


def get_template_alphafold(
    sum_length,
    paste_locs_list,
    read_pdb_index,
    motif_template,
    origin_motif_locs,
    motif_locs_chain_list,
    motif_pdb_path,
):
    template_all_atom_positions = np.zeros((1, sum_length, 37, 3))
    template_all_atom_masks = np.zeros((1, sum_length, 37))
    template_aatype = np.zeros((1, sum_length, 22))
    template_confidence_scores = np.full((1, 780), -1)

    for paste_loc, pdb_index, template_stat, origin_motif_loc, motif_loc_chain in zip(
        paste_locs_list,
        read_pdb_index,
        motif_template,
        origin_motif_locs,
        motif_locs_chain_list,
    ):
        if template_stat:
            motif_seq, motif_template_atoms, motif_atom_masks = read_pdb(
                motif_pdb_path[pdb_index],
                origin_motif_loc,
                motif_loc_chain,
                "alphafold",
            )
            try:
                template_all_atom_positions[:, paste_loc[0] : paste_loc[1], :, :] = (
                    motif_template_atoms
                )
            except:
                print(paste_loc)
                template_all_atom_positions[:, paste_loc[0] : paste_loc[1], :, :] = (
                    motif_template_atoms
                )
            template_all_atom_masks[:, paste_loc[0] : paste_loc[1], :] = (
                motif_atom_masks
            )
            for idx, aatype in zip(range(paste_loc[0], paste_loc[1]), motif_seq):
                template_aatype[:, idx, HHBLITS_AA_TO_ID[aatype]] = 1

    template_features = {
        "template_all_atom_positions": template_all_atom_positions,
        "template_all_atom_masks": template_all_atom_masks,
        "template_sequence": [f"none".encode()],
        "template_aatype": template_aatype,
        "template_confidence_scores": template_confidence_scores,
        "template_domain_names": [f"none".encode()],
        "template_release_date": [f"none".encode()],
    }

    return template_features


def get_aa_freq(AA_freq_file, exclude_aa, alphabet):
    if AA_freq_file is not None:
        AA_freq = json.load(open(AA_freq_file, "rb"))

    for aa in exclude_aa:
        if aa < len(alphabet):
            if alphabet[aa] in AA_freq:
                del AA_freq[alphabet[aa]]

    # Re-compute frequencies to sum to 1.
    sum_freq = np.sum(list(AA_freq.values()))
    adj_freq = [f / sum_freq for f in list(AA_freq.values())]
    aa_freq = dict(zip(AA_freq, adj_freq))

    return aa_freq


def get_modifiable_aa(sum_length, force_idx):
    mutable_aa = []
    for i in range(sum_length):
        if i not in force_idx:
            mutable_aa.append(i)

    return mutable_aa


def get_idx_unit_map(clean_code, length_dict):
    idx_unit_map = {}
    start_index = 0
    for unit in clean_code:
        unit_index = [i for i in range(start_index, start_index + length_dict[unit])]
        for idx in unit_index:
            idx_unit_map[idx] = unit
        start_index += length_dict[unit]
    return idx_unit_map
