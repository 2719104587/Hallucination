import copy
import torch
import numpy as np
from typing import List


def update_length_dict(length_dict, sample_type, change_unit):
    length_dict_copy = copy.deepcopy(length_dict)
    if sample_type == "mut":
        return length_dict
    elif sample_type == "add":
        length_dict_copy[change_unit] += 1
    elif sample_type == "pop":
        length_dict_copy[change_unit] -= 1
    return length_dict_copy


def update_loss_strategy(loss_strategy, sample_type, idxs):
    if sample_type == "mut":
        return loss_strategy
    else:
        loss_strategy_copy = copy.deepcopy(loss_strategy)
        for index, strategy in enumerate(loss_strategy):
            unit_interval = strategy[-1]
            for interval_index, interval in enumerate(unit_interval):
                start, end = interval[0], interval[1]
                for idx in idxs:
                    if start >= idx:
                        if sample_type == "add":
                            loss_strategy_copy[index][-1][interval_index][0] += 1
                        elif sample_type == "pop":
                            loss_strategy_copy[index][-1][interval_index][0] -= 1
                    if end > idx:
                        if sample_type == "add":
                            loss_strategy_copy[index][-1][interval_index][1] += 1
                        elif sample_type == "pop":
                            loss_strategy_copy[index][-1][interval_index][1] -= 1

        return loss_strategy_copy


def update_chain_info(chain_info, sample_type, change_unit):
    if sample_type == "mut":
        return chain_info
    else:
        chain_info_copy = copy.deepcopy(chain_info)
        for index, info in enumerate(chain_info):
            if change_unit == info[0]:
                if sample_type == "add":
                    chain_info_copy[index][1] += 1
                elif sample_type == "pop":
                    chain_info_copy[index][1] -= 1

        return chain_info_copy


def update_force_idx(force_idx, sample_type, idxs):
    if sample_type == "mut":
        return force_idx
    else:
        force_idx_copy = copy.deepcopy(force_idx)
        for idx in idxs:
            for index, aa_idx in enumerate(force_idx):
                if aa_idx >= idx:
                    if sample_type == "add":
                        force_idx_copy[index] += 1
                    elif sample_type == "pop":
                        force_idx_copy[index] -= 1

        return force_idx_copy


def update_mutate_idx_list(mutate_idx_list, sample_type, idxs):
    if sample_type == "mut":
        return mutate_idx_list
    else:
        mutate_idx_list_copy = copy.deepcopy(mutate_idx_list)
        for sample_index, sample_idx in enumerate(idxs):
            if sample_type == "add":
                mutate_idx_list_copy = mutate_idx_list_copy[
                    0 : sample_idx + sample_index + 1
                ] + [i + 1 for i in mutate_idx_list_copy[sample_idx + sample_index :]]
            elif sample_type == "pop":
                mutate_idx_list_copy = mutate_idx_list_copy[
                    0 : sample_idx - sample_index
                ] + [
                    i - 1 for i in mutate_idx_list_copy[sample_idx - sample_index + 1 :]
                ]
        return mutate_idx_list_copy


def update_idx(idx, sample_type, idxs):
    if sample_type == "mut":
        return idx
    else:
        idx_list = idx.numpy().tolist()
        idx_list_copy = copy.deepcopy(idx_list)
        for sample_index, sample_idx in enumerate(idxs):
            if sample_type == "add":
                idx_list_copy = idx_list_copy[0 : sample_idx + sample_index + 1] + [
                    i + 1 for i in idx_list_copy[sample_idx + sample_index :]
                ]
            elif sample_type == "pop":
                idx_list_copy = idx_list_copy[0 : sample_idx - sample_index] + [
                    i - 1 for i in idx_list_copy[sample_idx - sample_index + 1 :]
                ]

        return torch.tensor(idx_list_copy)


def update_paste_locs_list(paste_locs_list, sample_type, idxs):
    if sample_type == "mut":
        return paste_locs_list
    else:
        paste_locs_list_copy = copy.deepcopy(paste_locs_list)
        for idx in idxs:
            for index, paste_loc in enumerate(paste_locs_list):
                if paste_loc[0] >= idx:
                    if sample_type == "add":
                        paste_locs_list_copy[index][0] += 1
                    elif sample_type == "pop":
                        paste_locs_list_copy[index][0] -= 1
                if paste_loc[1] > idx:
                    if sample_type == "add":
                        paste_locs_list_copy[index][1] += 1
                    elif sample_type == "pop":
                        paste_locs_list_copy[index][1] -= 1
        return paste_locs_list_copy


def update_is_motif(is_motif_list, sample_type, idxs):
    if sample_type == "mut":
        return is_motif_list
    else:
        is_motif_list_result = []
        for is_motif in is_motif_list:
            is_motif_copy = is_motif.tolist()
            for sample_index, sample_idx in enumerate(idxs):
                if sample_type == "add":
                    is_motif_copy.insert(sample_idx + sample_index, False)
                elif sample_type == "pop":
                    is_motif_copy.pop(sample_idx - sample_index)
            is_motif_list_result.append(np.array(is_motif_copy))

        return is_motif_list_result


def update_motif_gaps(sum_length: int, paste_locs_list: List[List[int]]):
    motif_gaps = []
    sign_index = 0
    for paste_loc in paste_locs_list:
        motif_gaps.append(paste_loc[0] - sign_index)
        sign_index = paste_loc[1]
    motif_gaps.append(sum_length - sign_index)
    return motif_gaps
