import os
from typing import List

from hallucination.model.alphafold2.residue_constants import chain_ids
from hallucination.utils import check_interval, merge_intervals


def parse_locs(locs):
    if locs is None:
        return None, None

    locs = [loc for loc in locs.split(",") if len(loc) > 0]
    chain_list = []
    loc_list = []
    for loc in locs:
        chain = loc[0]
        if chain.isalpha():
            loc = loc[1:]
        else:
            chain = "A"
        if "-" in loc:
            l, r = loc.split("-")
            l, r = int(l), int(r)
        else:
            l, r = int(loc), int(loc)
        check_interval([l, r])
        chain_list.append(chain)
        loc_list.append([l, r])

    return chain_list, loc_list


def parse_locs_merge(locs):
    if locs is None:
        return None

    locs = [loc for loc in locs.split(",") if len(loc) > 0]

    result_functional_sites = []
    for loc in locs:
        chain = loc[0]
        if chain.isalpha():
            loc = loc[1:]
        else:
            chain = "A"
        if "-" in loc:
            l, r = loc.split("-")
            l, r = int(l), int(r)
        else:
            l, r = int(loc), int(loc)
        check_interval([l, r])
        result_functional_sites += [f"{chain}{i}" for i in range(l, r + 1)]

    return result_functional_sites


def parse_cluster(cluster_str: str, val_num: int):
    if cluster_str is None:
        return None
    cluster_split = [i for i in cluster_str.split(",") if len(i) > 0]

    if len(cluster_split) != val_num:
        raise ValueError("cluster_str must have the same length as val_num")

    temp_cluster = {}
    for idx, unit in enumerate(cluster_split):
        if unit not in temp_cluster:
            temp_cluster[unit] = []
        temp_cluster[unit].append(idx)

    return temp_cluster


def pp_motif_args(
    motif_locs,
    motif_truth,
    motif_cluster,
    relevance,
    motif_template,
    special_motif_index,
    special_motif_extra_seqs,
    motif_gaps,
    motif_gaps_cluster,
    paste_locs,
    functional_sites,
    motif_dir,
    prefix,
):
    if motif_locs is None:
        return [None for i in range(12)]

    ## process motif_locs
    motif_locs_chain_list, motif_locs_list = parse_locs(motif_locs)

    ## process motif_truth
    if motif_truth is None:
        motif_with_pdb_index = [0] * len(motif_locs_chain_list)
    else:
        motif_with_pdb_index = [
            int(i) - 1 for i in motif_truth.split(",") if len(i) > 0
        ]

    ## process motif_cluster
    temp_motif_cluster = {}
    if motif_cluster is None:
        if relevance:
            temp_motif_cluster = {"A": [i for i in range(len(motif_locs_list))]}
        else:
            for i in range(len(motif_locs_list)):
                temp_motif_cluster[chain_ids[i]] = [i]
    else:
        temp_motif_cluster = parse_cluster(motif_cluster, len(motif_locs_list))

    motif_cluster = temp_motif_cluster

    if motif_template is None:
        motif_template = [False] * len(motif_locs_chain_list)
    else:
        temp_motif_template = []
        for i in motif_template.split(","):
            if i == "T":
                temp_motif_template.append(True)
            elif i == "F":
                temp_motif_template.append(False)
            else:
                raise ValueError("motif_template must be T or F")
        motif_template = temp_motif_template

    if special_motif_index is None or special_motif_extra_seqs is None:
        special_motif_index = []
        special_motif_extra_seqs = []
        special_motif_expand = []
    else:
        special_motif_index = [
            int(i) for i in special_motif_index.split(",") if len(i) > 0
        ]
        temp_special_motif_extra_seqs = []
        temp_special_motif_expand = []
        for extra_seqs in special_motif_extra_seqs:
            extra_seqs = [i for i in extra_seqs.split(",")]
            if len(extra_seqs) != 2:
                raise ValueError("special_motif_extra_seqs must be like seq1,seq2")
            temp_special_motif_extra_seqs.append(extra_seqs)
            temp_special_motif_expand.append([len(extra_seqs[0]), len(extra_seqs[1])])
        special_motif_extra_seqs = temp_special_motif_extra_seqs
        special_motif_expand = temp_special_motif_expand

        if len(special_motif_index) != len(special_motif_extra_seqs) or len(
            special_motif_index
        ) != len(special_motif_expand):
            raise ValueError(
                "special_motif_index and special_motif_extra_seqs must have the same length"
            )

    ## process motif_gap
    if motif_gaps is not None:
        motif_gaps_split = [
            gap_str for gap_str in motif_gaps.split(",") if len(gap_str) > 0
        ]

        temp_motif_gaps = []
        for gap_str in motif_gaps_split:
            gap_split = [i for i in gap_str.split("-") if len(i) > 0]
            if len(gap_split) == 2:
                l, r = int(gap_split[0]), int(gap_split[1])
            elif len(gap_split) == 1:
                l = r = int(gap_split[0])
            else:
                raise TypeError("motif_gap must be like 1-10,10-20")
            check_interval([l, r])
            temp_motif_gaps.append([l, r])

        if motif_gaps_cluster is not None:
            motif_gaps_cluster = parse_cluster(
                motif_gaps_cluster, len(motif_gaps_split)
            )

            updated_motif_gaps = {}
            for cluster_id, motif_idx_list in motif_gaps_cluster.items():
                intervals = [temp_motif_gaps[idx] for idx in motif_idx_list]
                l, r = merge_intervals(intervals)

                for idx in motif_idx_list:
                    updated_motif_gaps[idx] = [l, r]

            temp_motif_gaps = [
                updated_motif_gaps[idx] for idx in range(len(temp_motif_gaps))
            ]

        else:
            motif_gaps_cluster = None

        motif_gaps = temp_motif_gaps

    else:
        motif_gaps_cluster = None

    ## process paste_locs
    if paste_locs is not None:
        _, paste_locs = parse_locs(paste_locs)

    ## process functional_sites
    functional_sites = parse_locs_merge(functional_sites)

    if len(special_motif_index) != len(special_motif_expand):
        raise TypeError(
            "special_motif_index and special_motif_expand must have the same length"
        )

    motif_pdb_path = [
        os.path.join(motif_dir, i + ".pdb") for i in prefix.split(",") if len(i) > 0
    ]

    if max(motif_with_pdb_index) + 1 > len(motif_pdb_path):
        raise ValueError(
            "motif_with_pdb_index must be less than the number of motif_pdb_path"
        )

    return (
        motif_locs_chain_list,
        motif_locs_list,
        motif_with_pdb_index,
        motif_cluster,
        motif_template,
        special_motif_index,
        special_motif_extra_seqs,
        special_motif_expand,
        motif_gaps,
        motif_gaps_cluster,
        paste_locs,
        functional_sites,
        motif_pdb_path,
    )


def pp_target_args(
    motif_target: str, target_pdbs: str, target_chains: str, combine_positions: List
):
    if motif_target is None:
        return None, None, None, None, None
    else:
        ## process motif_target
        temp_motif_target = []
        for i in motif_target.split(","):
            if i == "T":
                temp_motif_target.append(True)
            elif i == "F":
                temp_motif_target.append(False)
            else:
                raise ValueError("motif_target must be T or F")
        motif_target = temp_motif_target

        ## process target_pdbs
        target_pdbs = [
            pdb_path
            for pdb_path in target_pdbs.split(",")
            if len(pdb_path) > 0 and os.path.isfile(pdb_path)
        ]

        ## process target_chains
        target_chains = [
            list(chain_str)
            for chain_str in target_chains.split(",")
            if len(chain_str) > 0
        ]

        ## process combine_positions
        combine_locs_list = []
        combine_locs_chain_list = []
        for locs_str in combine_positions:
            chain_list, loc_list = parse_locs(locs_str)
            combine_locs_list.append(loc_list)
            combine_locs_chain_list.append(chain_list)

        if (
            len(target_pdbs)
            == len(target_chains)
            == len(combine_locs_list)
            == len(combine_locs_chain_list)
        ):
            return (
                motif_target,
                target_pdbs,
                target_chains,
                combine_locs_chain_list,
                combine_locs_list,
            )
        else:
            raise ValueError(
                "motif_target, target_pdbs, target_chains, combine_positions must have the same length"
            )
