import logging
import math
import os
from typing import Tuple
import click
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax

import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hallucination.config.default_config import get_default_config
from hallucination.motif_reader import read_motif

from hallucination.init_func import (
    convert_msa_from_letter_to_int,
    get_length_tuple,
    get_length_dict,
    get_same_code_aa_map,
    get_loss_strategy,
    get_chain_info,
    get_idx,
    get_motif_gaps,
    get_motif_gaps_aa_map,
    get_unit_index,
    get_paste_locs,
    get_mutate_idx_list,
    get_aa_freq,
    get_target_ccan,
    get_template_trfold,
    get_template_alphafold,
    get_force_info,
    get_modifiable_aa,
    get_idx_unit_map,
)
from hallucination.update_indel import (
    update_chain_info,
    update_length_dict,
    update_idx,
    update_paste_locs_list,
    update_loss_strategy,
    update_force_idx,
    update_mutate_idx_list,
    update_is_motif,
    update_motif_gaps,
)

from hallucination.model.trfold.predict import (
    trfold_predict,
    logits_to_probs,
    load_struture_module,
)
from hallucination.model.alphafold2.model import setup_models
from hallucination.model.alphafold2.predict import af2_predict

from hallucination.loss import hl_score
from hallucination.optimizer import NSGD

from hallucination.utils import (
    read_fasta,
    to_pdb,
    write_pickle,
    id2aa,
    alpha_1,
    is_subset,
    remove_non_uppercase,
    get_msa_one_hot,
)

import warnings

warnings.filterwarnings("ignore")


class HallucinateCore:
    def __init__(
        self,
        config_file_path,
        *,
        device,
        output_dir,
        motif_pk_path,
        inpaint,
        mcmc_indel: bool,
        prefix,
        code: str,
        length_tuple: tuple,
        mutate_idx: str,
        AA_freq_file: str,
        sampler_type="random",
        predict_model="trfold",
        init_method="random",
        init_sequence=None,
        init_mode="both",
    ):
        self.logger = logging.getLogger("hallucination")
        # cfg
        self.cfg = get_default_config()
        if config_file_path is not None and os.path.exists(config_file_path):
            self.cfg.merge_from_file(config_file_path)
            self.logger.info(f"config loaded from {config_file_path}")

        # init
        self.sampler_type = sampler_type
        self.predict_model = predict_model
        self.output_dir = output_dir
        self.prefix = prefix
        self.motif_pk_path = motif_pk_path
        self.inpaint = inpaint
        self.mcmc_indel = mcmc_indel
        self.init_mode = init_mode

        # runtime option
        self.device = device
        assert self.device in ["cpu", "cuda"]
        if self.predict_model == "alphafold":
            self.alpha_model = setup_models(
                data_dir=self.cfg.alphafold.data_dir,
                model_id=self.cfg.alphafold.model_id,
                recycles=self.cfg.alphafold.recycles,
                msa_clusters=self.cfg.alphafold.msa_clusters,
            )

        if self.predict_model == "trfold" or self.init_mode in ["both", "gd"]:
            self.trfold_model = load_struture_module(
                self.cfg.trfold.ckpt_path, self.device
            )
            if self.cfg.trfold.bf16:
                self.trfold_model = self.trfold_model.bfloat16()

        self.init_method = init_method
        self.init_sequence = init_sequence  # path to fasta file

        (
            self.motif_msa_list,
            self.motif_atom14_coord,
            self.motif_pdb_list,
            self.motif_with_pdb_index,
            self.motif_locs_list,
            self.motif_locs_chain_list,
            self.motif_cluster,
            self.motif_template,
            self.special_motif_index,
            self.special_motif_expand,
            self.expand_motif_loc_list,
            self.expand_motif_msa_list,
            motif_gaps,
            self.motif_gaps_cluster,
            paste_locs,
            self.functional_sites,
            self.motif_target,
            target_pdbs,
            target_chains,
            self.contig_locs_list,
            contig_locs_chain_list,
        ) = read_motif(self.motif_pk_path, logger=self.logger)

        ## 确定gap长度
        self.motif_gaps = get_motif_gaps(
            motif_gaps, self.motif_gaps_cluster
        )  ## indel change
        ## There are two kinds of amino acid association, one is gap association and the other is symmetric association(same code association).
        self.motif_gaps_aa_map = get_motif_gaps_aa_map(
            self.motif_locs_list, self.motif_gaps, self.motif_gaps_cluster
        )  ## index change

        if self.motif_gaps is not None:
            self.logger.debug(f"motif_gaps: {self.motif_gaps}")
            self.logger.debug(f"motif_gaps_aa_map: {self.motif_gaps_aa_map}")

        self.length_tuple = get_length_tuple(
            length_tuple, self.motif_locs_list, self.motif_gaps
        )

        self.code = code
        self.clean_code = remove_non_uppercase(self.code)
        self.length_dict = get_length_dict(
            self.clean_code, self.length_tuple
        )  # indel change

        self.same_code_aa_map = get_same_code_aa_map(
            self.clean_code, self.length_dict
        )  ## indel change

        self.loss_strategy = get_loss_strategy(
            self.code, self.clean_code, self.length_dict
        )  # indel change
        self.chain_info = get_chain_info(self.code, self.length_dict)  # indel change
        self.sum_length = sum([info[1] for info in self.chain_info])  # indel change

        self.logger.debug(f"length_tuple: {self.length_tuple}")
        self.logger.debug(f"clean_code: {self.clean_code}")
        self.logger.debug(f"length_dict: {self.length_dict}")
        self.logger.debug(f"same_code_aa_map: {self.same_code_aa_map}")
        self.logger.debug(f"loss_strategy: {self.loss_strategy}")
        self.logger.debug(f"chain_info: {self.chain_info}")
        self.logger.debug(f"sum_length: {self.sum_length}")

        self.force_aa = []
        self.force_idx = []  # indel change
        self.force_dict = {}
        self.exclude_aa = []
        self.aa_freq = get_aa_freq(AA_freq_file, self.exclude_aa, alpha_1)

        self.target_coord = []
        if self.motif_target is not None:
            for target_pdb, target_chain, contig_loc, contig_chain in zip(
                target_pdbs,
                target_chains,
                self.contig_locs_list,
                contig_locs_chain_list,
            ):
                target_xyz_ccan, contig_xyz_ccan = get_target_ccan(
                    target_pdb, target_chain, contig_loc, contig_chain
                )
                self.target_coord.append([target_xyz_ccan, contig_xyz_ccan])

        self.idx = get_idx(self.code, self.length_dict)  # indel change

        self.paste_locs_list = get_paste_locs(
            self.motif_locs_list,
            self.motif_gaps,
            self.length_tuple,
            paste_locs,
            self.chain_info,
            self.sum_length,
            self.special_motif_index,
            self.special_motif_expand,
        )  # indel change

        self.logger.debug(f"paste_locs_list: {self.paste_locs_list}")
        ## is_motif_list: indel change
        self.is_motif_list, self.motif_truth_list, self.motif_aatype_list = (
            self.prepare_motif_features()
        )
        self.force_dict, self.motif_force_dict = get_force_info(
            self.clean_code,
            self.length_dict,
            self.force_idx,  ##  non-null
            self.force_aa,  ##  non-null
            self.paste_locs_list,
            self.special_motif_index,
            self.special_motif_expand,
        )

        self.logger.debug(f"force_idx: {self.force_idx}")
        self.logger.debug(f"foce_aa: {[id2aa(aa) for aa in self.force_aa]}")

        self.can_change_idx = get_modifiable_aa(
            self.sum_length, self.force_idx
        )  ## indel change

        self.idx_unit_map = get_idx_unit_map(
            self.clean_code, self.length_dict
        )  ## indel change

        self.logger.debug(f"can_change_idx: {self.can_change_idx}")
        self.logger.debug(f"idx_unit_map: {self.idx_unit_map}")

        # indel change

        if self.init_method in ["external_sequence", "external_both"]:
            qid, self.sequence = read_fasta(self.init_sequence)
        elif self.init_method in ["random", "external_template"]:
            self.sequence = "".join(self.programming_seq_init())

        if self.init_method in ["external_template", "external_both"]:
            if self.predict_model == "trfold":
                self.template = get_template_trfold(
                    self.sum_length,
                    self.paste_locs_list,
                    self.motif_with_pdb_index,
                    self.motif_template,
                    self.motif_locs_list,
                    self.motif_locs_chain_list,
                    self.motif_pdb_list,
                )
            else:
                self.template = get_template_alphafold(
                    self.sum_length,
                    self.paste_locs_list,
                    self.motif_with_pdb_index,
                    self.motif_template,
                    self.motif_locs_list,
                    self.motif_locs_chain_list,
                    self.motif_pdb_list,
                )
        else:
            if self.predict_model == "trfold":
                self.template = torch.zeros(
                    1, self.sum_length, self.sum_length, 82
                )  # indel change
            else:
                self.template = None

        if self.init_method != "random":
            self.unit_index = get_unit_index(
                self.clean_code, self.length_dict, self.init_method, self.sequence
            )  # GD use

        ## The parameter specifies the interval within which the amino acid type can be changed
        self.mutate_idx_list = get_mutate_idx_list(
            mutate_idx, self.paste_locs_list
        )  ## indel change

        self.logger.debug(f"mutate_idx_list: {self.mutate_idx_list}")

        self.__used = False  # a instance can be run for ONLY ONCE

    def prepare_motif_features(self):
        if len(self.motif_locs_list) > 0:
            motif_truth_list = []
            motif_aatype_list = []
            is_motif_list = []

            for cluster_id, motif_idx_list in self.motif_cluster.items():
                motif_truth = np.zeros((self.sum_length, 14, 3), dtype=np.float32)
                motif_aatype = np.full(self.sum_length, 50, dtype=int)
                is_motif = np.full(self.sum_length, False)

                for motif_idx in motif_idx_list:
                    paste_s, paste_e = self.paste_locs_list[motif_idx]
                    expand_motif_loc = self.expand_motif_loc_list[motif_idx]
                    assert paste_e - paste_s == self.motif_msa_list[motif_idx].shape[0]
                    is_motif[paste_s:paste_e] = True
                    motif_truth[paste_s:paste_e] = self.motif_atom14_coord[motif_idx]
                    motif_aatype[paste_s:paste_e] = self.motif_msa_list[motif_idx]
                    if motif_idx in self.special_motif_index:
                        expand_f = self.special_motif_expand[
                            self.special_motif_index.index(motif_idx)
                        ][0]
                        expand_b = self.special_motif_expand[
                            self.special_motif_index.index(motif_idx)
                        ][1]
                        start = paste_s - expand_f
                        end = paste_e + expand_b
                    else:
                        start = paste_s
                        end = paste_e

                    assert end - start == len(expand_motif_loc)

                    for idx, (paste_idx, ori_motif_aa) in enumerate(
                        zip(range(start, end), expand_motif_loc)
                    ):
                        if ori_motif_aa in self.functional_sites:
                            self.force_idx.append(paste_idx)
                            self.force_aa.append(
                                self.expand_motif_msa_list[motif_idx][idx]
                            )

                motif_truth = motif_truth[is_motif, :, :].copy()
                motif_aatype = motif_aatype[is_motif].copy()
                motif_truth = torch.from_numpy(motif_truth).float().cuda().unsqueeze(0)
                motif_aatype = torch.from_numpy(motif_aatype).long().cuda().unsqueeze(0)
                motif_truth_list.append(motif_truth)
                motif_aatype_list.append(motif_aatype)
                is_motif_list.append(is_motif)

        else:
            is_motif_list = []
            motif_truth_list = []
            motif_aatype_list = []

        return is_motif_list, motif_truth_list, motif_aatype_list

    def initialize_logits(self, length, force_idx, force_aa, sequence) -> torch.Tensor:
        n_cat = 22
        self.exclude_aa = [20, 21]
        if self.init_method == "random":
            input_logits = np.random.normal(
                loc=0.0, scale=1.0, size=(1, 1, length, n_cat)
            )
            # input_logits[:, :, 24:24+self.gap, :] = input_logits[:, :, 48:48+self.gap, :]
            input_logits = torch.from_numpy(input_logits).float()
        else:
            msa = [sequence]
            msa = convert_msa_from_letter_to_int(msa)
            input_logits = (
                F.one_hot(torch.from_numpy(msa).long(), n_cat).unsqueeze(0).float()
            )
            self.logger.debug(input_logits.shape)
        bias = torch.zeros_like(input_logits)
        for aa in self.exclude_aa:
            bias[0, 0, :, aa] = bias[0, 0, :, aa] - 1e9
        for i, aa in zip(force_idx, force_aa):
            bias[0, 0, i, aa] = bias[0, 0, i, aa] + 1e18
        input_logits = input_logits + bias
        return input_logits

    def gradient_descent(
        self,
        steps: int,
    ):
        is_motif, motif_truth, motif_aatype = (
            self.is_motif_list,
            self.motif_truth_list,
            self.motif_aatype_list,
        )

        model = self.trfold_model
        model.train()
        initialize_logits_dict = {}
        unit_list = sorted(self.length_dict.keys())
        for unit, length in self.length_dict.items():
            if self.init_method == "random":
                input_logits = self.initialize_logits(
                    length,
                    self.force_dict[unit]["force_idx"],
                    self.force_dict[unit]["force_aa"],
                    "",
                )
            else:
                unit_sequence = self.sequence[
                    self.unit_index[unit][0][0] : self.unit_index[unit][0][1]
                ]
                input_logits = self.initialize_logits(
                    length,
                    self.force_dict[unit]["force_idx"],
                    self.force_dict[unit]["force_aa"],
                    unit_sequence,
                )
            input_logits = input_logits.to(self.device)
            initialize_logits_dict[unit] = input_logits

        input_logits = torch.cat(
            [initialize_logits_dict[unit] for unit in unit_list], dim=2
        )

        self.logger.debug("Starting gradient descent...")
        #    prefix=str(args.paste_locs[0][0])+'_'+str(args.length)
        trb = {"out_prefix": "test", "step": -1, "msa": None, "loss_tot": 1e9}
        torch.set_grad_enabled(True)
        #    if args.drop > 0:
        #        Net = enable_dropout(Net)
        N, D, L, C = input_logits.shape
        # turn off gradients to make in-place modifications
        input_logits = input_logits.requires_grad_(True)
        if self.cfg.opt.optimizer == "nsgd":
            optimizer = NSGD(
                [input_logits], lr=self.cfg.opt.learning_rate * np.sqrt(L), dim=[-1, -2]
            )
        elif self.cfg.opt.optimizer == "adam":
            optimizer = torch.optim.Adam([input_logits], lr=self.cfg.opt.learning_rate)
        else:
            raise AssertionError("Unknown optimization, nsgd or adam")

        def lr_lambda(current_step: int):
            return 1 - 0.9 * current_step / 600.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1, verbose=False
        )
        early_stop = False
        for step in range(steps + 1):  # 1 extra iteration to compute final outputs
            optimizer.zero_grad()
            logits_dict = {}
            start_index = 0
            for unit in unit_list:
                logits_dict[unit] = input_logits[
                    :, :, start_index : start_index + self.length_dict[unit], :
                ]
                start_index += self.length_dict[unit]
            # no update on last iteration
            if step == steps:
                torch.set_grad_enabled(False)
            # force / exclude AAs
            input_logits_update = torch.cat(
                [logits_dict[unit] for unit in self.clean_code], dim=2
            )
            bias = torch.zeros_like(input_logits_update)
            for aa in self.exclude_aa:
                bias[0, 0, :, aa] = bias[0, 0, :, aa] - 1e9
            for i, aa in zip(self.force_idx, self.force_aa):
                bias[0, 0, i, aa] = bias[0, 0, i, aa] + 1e18
            input_logits_biased = input_logits_update + bias
            # gumbel-softmax sampling
            msa_one_hot = logits_to_probs(
                input_logits_biased, add_gumbel_noise=False, output_type="hard"
            )
            profile_one_hot = msa_one_hot.clone()
            cluster = torch.cat([msa_one_hot, profile_one_hot], dim=-1)
            extra = torch.argmax(msa_one_hot, dim=-1).detach()
            aatype = extra.clone().detach().squeeze(1)

            template = self.template.unsqueeze(0).cuda()

            idx = self.idx.unsqueeze(0).cuda()
            model.recycle_query = torch.zeros(
                (1, self.sum_length, 256), device=self.device
            )
            model.recycle_rr = torch.zeros(
                (1, self.sum_length, self.sum_length, 128), device=self.device
            )
            model.recycle_dist = torch.zeros(
                (1, self.sum_length, self.sum_length, 15), device=self.device
            )

            if self.cfg.trfold.bf16:
                template = template.bfloat16()
                model.recycle_query = model.recycle_query.bfloat16()
                model.recycle_rr = model.recycle_rr.bfloat16()
                model.recycle_dist = model.recycle_dist.bfloat16()
                cluster = cluster.bfloat16()

            num_recycle = 4
            for itr in range(num_recycle):
                res = model(cluster, extra, template, idx, aatype)
            # calculate loss
            net_out = {
                "distogram": res["distance"].to(self.device),
                "coord": res["atoms"].to(self.device),
                "mask": res["atom_mask"].to(self.device),
                "plddt": res["plddt"].to(self.device),
                "msa_one_hot": msa_one_hot,
                "ptm": res["ptm"].to(self.device),
            }

            # if self.init_method == "external_template":
            #     aatype = aatype.squeeze(0).detach().cpu()
            #     res_coord = torch.cat([res["atoms"][:, :, :3, :], res["atoms"][:, :, 4:5, :], res["atoms"][:, :, 3:4, :]], dim=2)
            #     self.template = get_template_feature(aatype, res_coord.detach().cpu()).cuda()

            hl_losses = []
            loss_total = 0
            loss_rmsd = 0
            for strategy in self.loss_strategy:
                hl_losses_part = hl_score(
                    net_out,
                    motif_truth,
                    is_motif,
                    motif_aatype,
                    self.motif_target,
                    self.target_coord,
                    self.paste_locs_list,
                    self.cfg,
                    self.clean_code,
                    strategy,
                    is_gd=True,
                    predict_model=self.predict_model,
                )
                hl_losses.append(hl_losses_part)
                if len(self.motif_locs_list) > 0 and "rmsd" in hl_losses_part.keys():
                    loss_rmsd = hl_losses_part["rmsd"]
                loss_total += hl_losses_part["tot"]
            # track intermediate losses
            if (
                self.cfg.track_step is not None
                and step % (1 * self.cfg.track_step) == 0
            ):
                sequence = np.squeeze(
                    torch.argmax(msa_one_hot.detach(), dim=-1).cpu().numpy()
                )
                for i, aa in zip(self.force_idx, self.force_aa):
                    assert id2aa(sequence[i]) == id2aa(aa)
                info_str = ""
                for node_id, hl_loss in enumerate(hl_losses):
                    info_str += f"  node_{node_id}  "
                    info_str += ", ".join(
                        [
                            f"{name}: {float(value):>6.4f}"
                            for name, value in hl_loss.items()
                        ]
                    )
                if step == 0:
                    trb["sequence"] = "".join([alpha_1[i] for i in sequence])
                    keys = []
                    for hl_loss in hl_losses:
                        keys.extend(hl_loss.keys())
                    self.logger.debug("  ".join(keys))
                info_str = f"{step}  loss_tot :{loss_total.item():>6.4f}  " + info_str
                self.logger.debug(info_str)

                ### debug save
                if False:
                    best_coords = torch.squeeze(res["atoms"].detach()).cpu().numpy()
                    grids = 0.01 + 0.02 * np.arange(50)
                    best_seq_plddt = np.sum(
                        grids
                        * softmax(
                            torch.squeeze(res["plddt"].detach()).cpu().numpy(), axis=-1
                        ),
                        axis=-1,
                    )
                    sequence = np.squeeze(
                        torch.argmax(msa_one_hot.detach(), dim=-1).cpu().numpy()
                    )
                    current_AA = "".join([alpha_1[i] for i in sequence])
                    # self.logger.debug(current_AA)
                    trial_num = 1
                    output_pdb_path = os.path.join(
                        self.output_dir,
                        f"{self.prefix}_{trial_num}_{len(current_AA)}_{step}.pdb",
                    )
                    output_fasta_path = os.path.join(
                        self.output_dir,
                        f"{self.prefix}_{trial_num}_{len(current_AA)}_0.fasta",
                    )
                    with open(output_fasta_path, "a", encoding="utf8") as fa:
                        fa.write(current_AA + "\n")
                    with open(output_pdb_path, "w") as f_out:
                        f_out.write(
                            to_pdb(
                                current_AA,
                                best_coords,
                                best_seq_plddt,
                                None,
                                self.chain_info,
                            )
                        )

            # best design so far
            if float(loss_total) < trb["loss_tot"]:
                trb["loss_tot"] = float(loss_total)
                trb["hl_losses"] = hl_losses
                for node_id, hl_loss in enumerate(hl_losses):
                    for name, value in hl_loss.items():
                        if f"node_{node_id}_loss_" + name not in trb.keys():
                            trb[f"node_{node_id}_loss_" + name] = 1e9
                        trb[f"node_{node_id}_loss_" + name] = float(value)
                sequence = np.squeeze(
                    torch.argmax(msa_one_hot.detach(), dim=-1).cpu().numpy()
                )
                trb["sequence"] = "".join([alpha_1[i] for i in sequence])
                trb["template"] = self.template
                trb["step"] = step
                trb["coords"] = torch.squeeze(res["atoms"].detach()).cpu().numpy()
                trb["masks"] = torch.squeeze(net_out["mask"].detach()).cpu().numpy()
                grids = 0.01 + 0.02 * np.arange(50)
                trb["plddt"] = np.sum(
                    grids
                    * softmax(
                        torch.squeeze(res["plddt"].detach()).cpu().numpy(), axis=-1
                    ),
                    axis=-1,
                )
                # early stop
                if (
                    trb["step"] > self.cfg.mcmc.earlystop
                    and float(loss_rmsd) < self.cfg.loss.earlystop_rmsd_thresh
                ):
                    early_stop = True
                    break
            if step != steps or early_stop:  # no update on last iteration
                loss_total.backward()
                optimizer.step()
                lr_scheduler.step()
        # final loss
        info_str = f"final : loss_tot :{trb['loss_tot']:>6.4f} "
        for node_id, hl_loss in enumerate(trb["hl_losses"]):
            info_str += ", ".join(
                [
                    f"  node_{node_id}_loss_{name} : {value.item():>6.4f}"
                    for name, value in hl_loss.items()
                ]
            )
        self.logger.debug(info_str)
        self.logger.debug(trb["sequence"])
        self.logger.debug(f"best loss step:{trb['step']}")

        return (
            trb["sequence"],
            trb["coords"],
            trb["plddt"],
            trb["loss_tot"],
            trb["masks"],
        )

    def uniform_sampler(
        self,
        sequences,
        num_changed_tokens,
        is_motif,
        motif_aatype,
        residue_plddt,
        exclude_aa=None,
        AA_freq=None,
        sample_freq=None,
    ):
        if exclude_aa is None:
            exclude_aa = ["-"]
        seq = list(sequences)

        ## Determine the sampling type of the round
        sample_type = np.random.choice(
            list(sample_freq.keys()), p=list(sample_freq.values())
        )

        if self.mutate_idx_list is not None and is_subset(
            self.can_change_idx, self.mutate_idx_list
        ):
            can_change_idx = self.mutate_idx_list
        else:
            can_change_idx = self.can_change_idx

        if self.sampler_type == "random":
            change_idxs = np.random.choice(
                np.array(can_change_idx, dtype=int),
                num_changed_tokens,
                replace=False,
            )
        else:
            plddt = residue_plddt[can_change_idx]
            w = np.absolute(plddt - np.max(plddt)) / np.sum(
                np.absolute(plddt - np.max(plddt))
            )
            change_idxs = np.random.choice(
                can_change_idx, num_changed_tokens, p=w.tolist()
            )

        change_idxs_list = []
        change_unit_list = []

        ## Symmetric designs with motif gaps are not considered
        for index in change_idxs:
            if self.motif_gaps_aa_map is not None:
                change_index = self.motif_gaps_aa_map[index]
            else:
                change_index = [index]
            for idx in change_index:
                change_idxs_list.append(self.same_code_aa_map[idx])
                change_unit_list.append(self.idx_unit_map[idx])

        update_dict = {}
        update_dict["length_dict"] = self.length_dict
        update_dict["loss_strategy"] = self.loss_strategy
        update_dict["chain_info"] = self.chain_info
        update_dict["force_idx"] = self.force_idx
        if self.mutate_idx_list is not None and is_subset(
            self.can_change_idx, self.mutate_idx_list
        ):
            update_dict["mutate_idx_list"] = self.mutate_idx_list
        else:
            update_dict["mutate_idx_list"] = None
        update_dict["idx"] = self.idx
        update_dict["is_motif"] = is_motif
        update_dict["paste_locs_list"] = self.paste_locs_list
        for change_unit, idxs in zip(change_unit_list, change_idxs_list):
            idxs = sorted(idxs)
            if sample_type == "mut":
                aa_mut = np.random.choice(
                    list(AA_freq.keys()), p=list(AA_freq.values())
                )
                for idx in idxs:
                    seq[idx] = aa_mut
            elif sample_type == "add":
                aa_add = np.random.choice(
                    list(AA_freq.keys()), p=list(AA_freq.values())
                )
                for count, idx in enumerate(idxs):
                    seq.insert(idx + count, aa_add)
            elif sample_type == "pop":
                for count, idx in enumerate(idxs):
                    seq.pop(idx - count)

            update_dict["length_dict"] = update_length_dict(
                update_dict["length_dict"], sample_type, change_unit
            )
            update_dict["loss_strategy"] = update_loss_strategy(
                update_dict["loss_strategy"], sample_type, idxs
            )
            update_dict["chain_info"] = update_chain_info(
                update_dict["chain_info"], sample_type, change_unit
            )
            update_dict["force_idx"] = update_force_idx(
                update_dict["force_idx"], sample_type, idxs
            )
            if self.mutate_idx_list is not None and is_subset(
                self.can_change_idx, self.mutate_idx_list
            ):
                update_dict["mutate_idx_list"] = update_mutate_idx_list(
                    update_dict["mutate_idx_list"], sample_type, idxs
                )
            else:
                update_dict["mutate_idx_list"] = None

            update_dict["idx"] = update_idx(update_dict["idx"], sample_type, idxs)
            update_dict["paste_locs_list"] = update_paste_locs_list(
                update_dict["paste_locs_list"], sample_type, idxs
            )

            update_dict["is_motif"] = update_is_motif(
                update_dict["is_motif"], sample_type, idxs
            )

        update_dict["sum_length"] = len(seq)
        update_dict["motif_gaps"] = update_motif_gaps(
            update_dict["sum_length"], update_dict["paste_locs_list"]
        )
        update_dict["motif_gaps_aa_map"] = get_motif_gaps_aa_map(
            self.motif_locs_list, update_dict["motif_gaps"], self.motif_gaps_cluster
        )
        update_dict["same_code_aa_map"] = get_same_code_aa_map(
            self.clean_code, update_dict["length_dict"]
        )
        update_dict["can_change_idx"] = get_modifiable_aa(
            update_dict["sum_length"], update_dict["force_idx"]
        )
        update_dict["idx_unit_map"] = get_idx_unit_map(
            self.clean_code, update_dict["length_dict"]
        )
        if self.init_method in ["external_template", "external_both"]:
            if self.predict_model == "trfold":
                update_dict["template"] = get_template_trfold(
                    update_dict["sum_length"],
                    update_dict["paste_locs_list"],
                    self.motif_with_pdb_index,
                    self.motif_template,
                    self.motif_locs_list,
                    self.motif_locs_chain_list,
                    self.motif_pdb_list,
                )
            elif self.predict_model == "alphafold":
                update_dict["template"] = get_template_alphafold(
                    update_dict["sum_length"],
                    update_dict["paste_locs_list"],
                    self.motif_with_pdb_index,
                    self.motif_template,
                    self.motif_locs_list,
                    self.motif_locs_chain_list,
                    self.motif_pdb_list,
                )
        else:
            if self.predict_model == "trfold":
                update_dict["template"] = torch.zeros(
                    1, update_dict["sum_length"], update_dict["sum_length"], 82
                )
                if self.cfg.trfold.bf16:
                    update_dict["template"] = update_dict["template"].bfloat16()
            elif self.predict_model == "alphafold":
                update_dict["template"] = None

        return "".join(seq), update_dict

    @staticmethod
    def make_fold_params(sequence):
        msa = [sequence]
        msa = convert_msa_from_letter_to_int(msa)
        msa = torch.from_numpy(msa).long()
        aatype = msa[0, :].clone()
        return msa, aatype

    @torch.no_grad()
    def mcmc(
        self,
        steps,
        init_aa,
    ):
        is_motif, motif_truth, motif_aatype = (
            self.is_motif_list,
            self.motif_truth_list,
            self.motif_aatype_list,
        )

        self.logger.debug("mcmc init AA     " + init_aa)
        for i, aa in zip(self.force_idx, self.force_aa):
            assert init_aa[i] == id2aa(aa)
        self.logger.debug(f"mutate_idx_list:{self.mutate_idx_list}")
        if self.mcmc_indel:
            sample_freq = {
                "mut": float(1 - self.cfg.mcmc.indel_prob),
                "add": float(self.cfg.mcmc.indel_prob * (2 / 3)),
                "pop": float(self.cfg.mcmc.indel_prob / 3),
            }
        else:
            sample_freq = {"mut": 1}

        # Allow batches larger than 1
        # Prevent LM sampler from mutating forced positions
        trb = {"out_prefix": "test", "step": -1, "msa": None, "loss_tot": 1e9}
        if self.force_aa is not None:
            pass
        self.logger.debug("Starting MCMC...")
        sf_0 = np.exp(
            np.log(0.5) / self.cfg.mcmc.mcmc_halflife
        )  # scaling factor / step
        current_AA = init_aa

        msa_one_hot = get_msa_one_hot(current_AA).to(self.device)

        if self.predict_model == "trfold":
            msa, aatype = self.make_fold_params(current_AA)
            res = trfold_predict(
                self.trfold_model,
                msa,
                self.template,
                self.idx,
                aatype,
                self.cfg.trfold.recycles,
                self.cfg.trfold.bf16,
            )
            net_out = {
                "distogram": res["distance"].to(self.device),
                "coord": res["atoms"].to(self.device),
                "mask": res["atom_mask"].to(self.device),
                "plddt": res["plddt"].to(self.device),
                "ptm": res["ptm"].to(self.device),
                "pae": 0,
                "msa_one_hot": msa_one_hot,
            }
            grids = 0.01 + 0.02 * np.arange(50)
            residue_plddt = np.sum(
                grids
                * softmax(
                    torch.squeeze(net_out["plddt"].detach()).cpu().numpy(), axis=-1
                ),
                axis=-1,
            )

        elif self.predict_model == "alphafold":
            res = af2_predict(
                current_AA,
                self.idx,
                self.alpha_model,
                np.random.randint(42),
                self.template,
            )
            net_out = {
                "distogram": torch.unsqueeze(
                    torch.tensor(np.array(res[0]["distogram"]["logits"])), dim=0
                ).to(self.device),
                "coord": torch.unsqueeze(
                    torch.tensor(
                        np.array(res[0]["structure_module"]["final_atom_positions"])
                    ),
                    dim=0,
                ).to(self.device),
                "mask": torch.unsqueeze(
                    torch.tensor(
                        np.array(res[0]["structure_module"]["final_atom_mask"])
                    ),
                    dim=0,
                ).to(self.device),
                "plddt": torch.tensor(np.array(res[0]["plddt"])).to(self.device),
                "ptm": torch.full(
                    (1, self.sum_length, self.sum_length, 64),
                    torch.tensor(res[0]["ptm"]),
                ).to(self.device),
                "pae": res[0]["predicted_aligned_error"],
                "msa_one_hot": msa_one_hot,
            }
            residue_plddt = np.array(res[0]["plddt"])

        E_0 = 0.0
        E_rmsd = 0.0
        hl_losses = []
        for strategy in self.loss_strategy:
            hl_losses_part = hl_score(
                net_out,
                motif_truth,
                is_motif,
                motif_aatype,
                self.motif_target,
                self.target_coord,
                self.paste_locs_list,
                self.cfg,
                self.clean_code,
                strategy,
                is_gd=False,
                predict_model=self.predict_model,
            )
            E_0 += float(hl_losses_part["tot"])
            if len(self.motif_locs_list) > 0 and "rmsd" in hl_losses_part.keys():
                E_rmsd = float(hl_losses_part["rmsd"])
            hl_losses.append(hl_losses_part)

        self.logger.debug(hl_losses)
        trb["sequence"] = current_AA
        trb["coords"] = torch.squeeze(net_out["coord"].detach()).cpu().numpy()
        trb["plddt"] = residue_plddt
        trb["loss_tot"] = E_0
        trb["masks"] = torch.squeeze(net_out["mask"].detach()).cpu().numpy()
        trb["hl_losses"] = hl_losses
        trb["length_dict"] = self.length_dict
        trb["loss_strategy"] = self.loss_strategy
        trb["chain_info"] = self.chain_info
        trb["sum_length"] = self.sum_length
        trb["force_idx"] = self.force_idx
        trb["idx"] = self.idx
        trb["paste_locs_list"] = self.paste_locs_list

        for step in range(steps):
            sf = sf_0**step
            mutated_AA, update_dict = self.uniform_sampler(
                current_AA,
                self.cfg.mcmc.num_mutated_tokens,
                is_motif,
                motif_aatype,
                residue_plddt,
                exclude_aa=self.exclude_aa,
                AA_freq=self.aa_freq,
                sample_freq=sample_freq,
            )
            msa, aatype = self.make_fold_params(mutated_AA)

            msa_one_hot = get_msa_one_hot(mutated_AA).to(self.device)

            if self.predict_model == "trfold":
                res = trfold_predict(
                    self.trfold_model,
                    msa,
                    update_dict["template"],
                    update_dict["idx"],
                    aatype,
                    self.cfg.trfold.recycles,
                    self.cfg.trfold.bf16,
                )
                net_out = {
                    "distogram": res["distance"].to(self.device),
                    "coord": res["atoms"].to(self.device),
                    "mask": res["atom_mask"].to(self.device),
                    "plddt": res["plddt"].to(self.device),
                    "ptm": res["ptm"].to(self.device),
                    "pae": 0,
                    "msa_one_hot": msa_one_hot,
                }
                grids = 0.01 + 0.02 * np.arange(50)
                plddt_step = np.sum(
                    grids
                    * softmax(
                        torch.squeeze(net_out["plddt"].detach()).cpu().numpy(), axis=-1
                    ),
                    axis=-1,
                )

            elif self.predict_model == "alphafold":
                res = af2_predict(
                    mutated_AA,
                    update_dict["idx"],
                    self.alpha_model,
                    np.random.randint(42),
                    self.template,
                )
                net_out = {
                    "distogram": torch.unsqueeze(
                        torch.tensor(np.array(res[0]["distogram"]["logits"])), dim=0
                    ).to(self.device),
                    "coord": torch.unsqueeze(
                        torch.tensor(
                            np.array(res[0]["structure_module"]["final_atom_positions"])
                        ),
                        dim=0,
                    ).to(self.device),
                    "mask": torch.unsqueeze(
                        torch.tensor(
                            np.array(res[0]["structure_module"]["final_atom_mask"])
                        ),
                        dim=0,
                    ).to(self.device),
                    "plddt": torch.tensor(np.array(res[0]["plddt"])).to(self.device),
                    "ptm": torch.full(
                        (1, self.sum_length, self.sum_length, 64),
                        torch.tensor(res[0]["ptm"]),
                    ).to(self.device),
                    "pae": res[0]["predicted_aligned_error"],
                    "msa_one_hot": msa_one_hot,
                }
                plddt_step = np.array(res[0]["plddt"])

            # if self.init_method == "external_template":
            #     template = get_template_feature(aatype, res["atoms"][:, :, :5, :].detach().cpu())

            E_1 = 0.0
            E_rmsd = 0.0
            hl_losses_1 = []
            for strategy in update_dict["loss_strategy"]:
                hl_losses_part = hl_score(
                    net_out,
                    motif_truth,
                    update_dict["is_motif"],
                    motif_aatype,
                    self.motif_target,
                    self.target_coord,
                    update_dict["paste_locs_list"],
                    self.cfg,
                    self.clean_code,
                    strategy,
                    is_gd=False,
                    predict_model=self.predict_model,
                )
                E_1 += float(hl_losses_part["tot"])
                if len(self.motif_locs_list) > 0 and "rmsd" in hl_losses_part.keys():
                    E_rmsd = float(hl_losses_part["rmsd"])
                hl_losses_1.append(hl_losses_part)
            T_acc = sf * self.cfg.mcmc.T_acc_0

            # track intermediate losses
            if (
                self.cfg.track_step is not None
                and step % (1 * self.cfg.track_step) == 0
            ):
                info_str = ""
                loss_tot = 0.0
                for node_id, hl_loss in enumerate(hl_losses_1):
                    for name, value in hl_loss.items():
                        if name == "tot":
                            loss_tot += float(value)
                    info_str += f"  node_{node_id}  "
                    info_str += ", ".join(
                        [
                            f"{name}: {float(value):>6.4f}"
                            for name, value in hl_loss.items()
                        ]
                    )
                info_str = f"{step}  loss_tot :{loss_tot:>6.4f}" + info_str
                self.logger.debug(info_str)
                if self.mcmc_indel:
                    self.logger.debug(f"{step}  sum_length :{self.sum_length}")

            if E_1 < E_0:
                acceptances = 1.0
            else:
                acceptances = np.clip(math.exp(-(E_1 - E_0) / T_acc), 0.0, 1.0)
            uniform_random = np.random.uniform(low=0.0, high=1.0)
            if uniform_random < acceptances:
                self.logger.debug(
                    f"Step {step}: change accepted despite not improving the loss >> LOSS {float(E_1):>6.4f} --> {float(E_0):>6.4f}"
                )
                E_0 = E_1
                current_AA = mutated_AA
                hl_losses = hl_losses_1

                self.length_dict = update_dict["length_dict"]
                self.loss_strategy = update_dict["loss_strategy"]
                self.chain_info = update_dict["chain_info"]
                self.force_idx = update_dict["force_idx"]
                self.mutate_idx_list = update_dict["mutate_idx_list"]
                self.idx = update_dict["idx"]
                self.paste_locs_list = update_dict["paste_locs_list"]
                is_motif = update_dict["is_motif"]
                self.sum_length = update_dict["sum_length"]
                self.motif_gaps = update_dict["motif_gaps"]
                self.motif_gaps_aa_map = update_dict["motif_gaps_aa_map"]
                self.same_code_aa_map = update_dict["same_code_aa_map"]
                self.can_change_idx = update_dict["can_change_idx"]
                self.idx_unit_map = update_dict["idx_unit_map"]
                self.template = update_dict["template"]
                residue_plddt = plddt_step

            else:
                self.logger.debug(
                    f"Step {step}: change rejected >> LOSS {float(E_1):>6.4f} !-> {float(E_0):>6.4f}"
                )

            # best design so far
            if E_0 < trb["loss_tot"]:
                trb["sequence"] = current_AA
                trb["hl_losses"] = hl_losses
                loss_tot = 0.0
                for node_id, hl_loss in enumerate(hl_losses):
                    for name, value in hl_loss.items():
                        if name == "tot":
                            loss_tot += float(value)
                        if f"node_{node_id}_loss_" + name not in trb.keys():
                            trb[f"node_{node_id}_loss_" + name] = 1e9
                        trb[f"node_{node_id}_loss_" + name] = float(value)
                trb["loss_tot"] = loss_tot
                trb["step"] = step
                trb["coords"] = torch.squeeze(net_out["coord"].detach()).cpu().numpy()
                trb["masks"] = torch.squeeze(net_out["mask"].detach()).cpu().numpy()
                grids = 0.01 + 0.02 * np.arange(50)
                trb["plddt"] = residue_plddt
                # early stop
                if (
                    trb["step"] > self.cfg.mcmc.earlystop
                    and E_rmsd < self.cfg.loss.earlystop_rmsd_thresh
                ):
                    break
            else:
                # early stop
                if step - trb["step"] > self.cfg.loss.earlystop_steps:
                    break
        info_str = f"final : loss_tot :{trb['loss_tot']:>6.4f} "
        for node_id, hl_loss in enumerate(trb["hl_losses"]):
            info_str += ", ".join(
                [
                    f"  node_{node_id}_loss_{name} : {float(value):>6.4f}"
                    for name, value in hl_loss.items()
                ]
            )
        self.logger.debug(info_str)
        self.logger.debug(trb["sequence"])
        self.logger.debug(f"best loss step:{trb['step']}")

        for i, aa in zip(self.force_idx, self.force_aa):
            assert trb["sequence"][i] == id2aa(aa)
        return (
            trb["sequence"],
            trb["coords"],
            trb["plddt"],
            trb["loss_tot"],
            trb["masks"],
        )

    def programming_seq_init(self):
        letter_unique = list(set(list(self.clean_code)))
        seq_dict = {}
        for letter in letter_unique:
            seq_dict[letter] = [
                alpha_1[np.random.randint(0, len(alpha_1) - 1)]
                for i in range(self.length_dict[letter])
            ]
        seq = []
        for letter in self.clean_code:
            seq.extend(seq_dict[letter])

        ## non-null
        for i, aa in zip(self.force_idx, self.force_aa):
            seq[i] = id2aa(aa)

        return seq

    def run(
        self,
        trial_num,
    ) -> Tuple[str, str, str]:
        if self.__used:
            self.logger.critical("instance is used, need to new another one")
            return
        self.__used = True
        self.logger.debug(self.prefix)

        if self.init_mode in ["both", "gd"]:
            ## gd does not support indel
            best_sequence, best_seq_coord, best_seq_plddt, best_loss, best_masks = (
                self.gradient_descent(
                    self.cfg.steps,
                )
            )
        else:
            best_sequence = self.sequence

        if self.init_mode in ["both", "mcmc"]:
            best_sequence, best_seq_coord, best_seq_plddt, best_loss, best_masks = (
                self.mcmc(
                    self.cfg.mcmc_steps,
                    best_sequence,
                )
            )

        ## save best result
        output_fasta_path = os.path.join(
            self.output_dir, f"{self.prefix}_{trial_num}_{self.sum_length}.fasta"
        )
        with open(
            output_fasta_path,
            "w",
        ) as f_fasta:
            start_idx = 0
            for chain_name, chain_length in self.chain_info:
                f_fasta.write(f">design_{chain_name}\n")
                f_fasta.write(best_sequence[start_idx : start_idx + chain_length])
                f_fasta.write("\n")
                start_idx += chain_length
        output_pdb_path = os.path.join(
            self.output_dir, f"{self.prefix}_{trial_num}_{self.sum_length}.pdb"
        )
        with open(output_pdb_path, "w") as f_out:
            f_out.write(
                to_pdb(
                    best_sequence,
                    best_seq_coord,
                    best_seq_plddt,
                    best_masks,
                    self.chain_info,
                    self.predict_model,
                )
            )
        output_pk_path = os.path.join(
            self.output_dir, f"{self.prefix}_{trial_num}_{self.sum_length}_rec.pk"
        )
        write_pickle(
            {
                "length": self.sum_length,
                "paste_locs": self.paste_locs_list,
                "fixed_locs": self.force_idx,
                "loss_tot": best_loss,
            },
            output_pk_path,
        )
        self.logger.info(f"hallucination {trial_num} done")
        return output_fasta_path, output_pdb_path, output_pk_path


@click.command("hallucination-generator")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option("--prefix", type=str, required=True)
@click.option("--sampler-type", type=str, default="random")
@click.option("--AA-freq-file", type=str, default="")
@click.option("--predict-model", type=str, default="trfold")
@click.option("--init-method", type=str, default="random")
@click.option("--init-sequence", type=str)
@click.option("--init-mode", type=str, default="both")
@click.option("--inpaint", type=bool, is_flag=True)
@click.option("--mcmc-indel", type=bool, is_flag=True)
@click.option("--output-dir", type=click.Path(dir_okay=True), required=True)
@click.option("--length-tuple", type=tuple, required=True)
@click.option("--mutate-idx", type=str, multiple=True)
@click.option(
    "--motif-pk-path",
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option("--trial-id", type=int, required=True)
@click.option("--gpu", type=bool, is_flag=True)
@click.option("--code", type=str, required=True)
def __click_main(
    config: str,
    prefix: str,
    sampler_type: str,
    aa_freq_file: str,
    predict_model: str,
    init_method: str,
    init_sequence: str,
    init_mode: str,
    inpaint: bool,
    mcmc_indel: bool,
    motif_pk_path: str,
    output_dir: str,
    length_tuple: tuple,
    trial_id: int,  # (trial_idx+init_trial) you should sum trial_idx and init_trial for this
    mutate_idx: str,
    gpu: bool,
    code: str,
):
    from pprint import pp

    length_tuple = tuple(
        [int(i) for i in "".join(list(length_tuple))[1:-1].split(",") if len(i) > 0]
    )
    pp(locals())

    if len(mutate_idx) != 0:
        mutate_idx = mutate_idx[0]
    if not os.path.exists(aa_freq_file):
        aa_freq_file = None
    __main(
        config_file_path=str(config),
        prefix=prefix,
        sampler_type=sampler_type,
        AA_freq_file=aa_freq_file,
        predict_model=predict_model,
        init_method=init_method,
        init_sequence=init_sequence,
        init_mode=init_mode,
        inpaint=inpaint,
        mcmc_indel=mcmc_indel,
        motif_pk_path=str(motif_pk_path),
        output_dir=output_dir,
        length_tuple=length_tuple,
        trial_id=trial_id,
        mutate_idx=mutate_idx,
        device="cuda" if gpu else "cpu",
        code=code,
    )


def __main(
    *,
    config_file_path: str,
    prefix: str,
    sampler_type: str,
    AA_freq_file: str,
    predict_model: str,
    init_method: str,
    init_sequence: str,
    init_mode: str,
    inpaint: bool,
    mcmc_indel: bool,
    motif_pk_path: str,
    output_dir: str,
    length_tuple: tuple,
    code: str,
    trial_id: int,  # (trial_idx+init_trial) you should sum trial_idx and init_trial for this
    mutate_idx: str,
    device: str = "cuda",
):
    hallucinate = HallucinateCore(
        config_file_path=config_file_path,
        prefix=prefix,
        sampler_type=sampler_type,
        AA_freq_file=AA_freq_file,
        predict_model=predict_model,
        init_method=init_method,
        init_sequence=init_sequence,
        init_mode=init_mode,
        device=device,
        mutate_idx=mutate_idx,
        output_dir=output_dir,
        motif_pk_path=motif_pk_path,
        inpaint=inpaint,
        mcmc_indel=mcmc_indel,
        code=code,
        length_tuple=length_tuple,
    )

    o_fasta_path, o_pdb_path, o_pk_path = hallucinate.run(
        trial_id,
    )
    return o_fasta_path, o_pdb_path, o_pk_path


if __name__ == "__main__":
    __click_main()
