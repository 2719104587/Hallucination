import numpy as np
from Bio.PDB import PDBParser
import os

import hallucination.model.alphafold2.residue_constants as rc
from hallucination.init_func import convert_seq_from_letter_to_int
from hallucination.utils import load_pickle


def is_motif(idx, motif_loc):
    if len(motif_loc) == 1:
        if idx == motif_loc[0]:
            return True
        else:
            return False

    elif len(motif_loc) == 2:
        if idx >= motif_loc[0] and idx <= motif_loc[1]:
            return True
        else:
            return False


def load_motif(
    motif_pdb_list,
    motif_locs_list,
    motif_locs_chain_list,
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
    motif_target,
    target_pdbs,
    target_chains,
    contig_locs_list,
    contig_locs_chain_list,
):
    if motif_with_pdb_index is None:
        motif_with_pdb_index = [0 for i in len(motif_locs_list)]

    parser = PDBParser()

    ## motif sequence
    motif_msa_list = []
    expand_motif_msa_list = []
    ## motif_aa atomic coordinates
    motif_atom14_coord = []

    expand_motif_loc_list = []
    for motif_index, (motif_loc, chain_id, pdb_index) in enumerate(
        zip(motif_locs_list, motif_locs_chain_list, motif_with_pdb_index)
    ):
        atom14_coords = []
        motif_seqs = []
        ## If there are amino acids located at functional sites in the expand motif, the amino acid type will be conserved
        if special_motif_index is not None and motif_index in special_motif_index:
            expand_f = special_motif_expand[special_motif_index.index(motif_index)][0]
            expand_b = special_motif_expand[special_motif_index.index(motif_index)][1]
            if len(motif_loc) == 1:
                expand_motif_loc = (motif_loc[0] - expand_f, motif_loc[0] + expand_b)

            elif len(motif_loc) == 2:
                expand_motif_loc = (motif_loc[0] - expand_f, motif_loc[1] + expand_b)
            motif_extra_seqs = special_motif_extra_seqs[
                special_motif_index.index(motif_index)
            ]
        else:
            expand_motif_loc = motif_loc
            motif_extra_seqs = ["", ""]

        expand_motif_loc_list.append(
            [
                f"{chain_id}{i}"
                for i in range(expand_motif_loc[0], expand_motif_loc[1] + 1)
            ]
        )

        protein = parser.get_structure("X", motif_pdb_list[pdb_index])
        chains = {chain.id: chain for chain in protein.get_chains()}
        chain = chains[chain_id]
        for res in chain.get_residues():
            res_id = res.id[1]
            if is_motif(res_id, motif_loc):
                resname = res.get_resname()
                motif_seqs.append(rc.restype_3to1[resname])
                atoms = rc.residue_atoms[resname]
                res_atom = np.zeros((14, 3))
                for atom in atoms:
                    atom_idx = rc.restype_name_to_atom14_names[resname].index(atom)
                    if res.has_id(atom):
                        res_atom[atom_idx, :] = res[atom].get_coord()
                atom14_coords.append(res_atom)

        expand_motif_seqs = (
            list(motif_extra_seqs[0]) + motif_seqs + list(motif_extra_seqs[1])
        )

        motif_atom14_coord.append(np.stack(atom14_coords, axis=0))
        motif_msa_list.append(convert_seq_from_letter_to_int(motif_seqs))
        ## if special_motif_index is None, expand_motif_msa_list is same to motif_msa_list
        expand_motif_msa_list.append(convert_seq_from_letter_to_int(expand_motif_seqs))
    return {
        "motif_msa_list": motif_msa_list,
        "motif_atom14_coord_list": motif_atom14_coord,
        "motif_pdb_list": motif_pdb_list,
        "motif_with_pdb_index": motif_with_pdb_index,
        "motif_locs_list": motif_locs_list,
        "motif_locs_chain_list": motif_locs_chain_list,
        "motif_cluster": motif_cluster,
        "motif_template": motif_template,
        "special_motif_index": special_motif_index,
        "special_motif_expand": special_motif_expand,
        "expand_motif_loc_list": expand_motif_loc_list,
        "expand_motif_msa_list": expand_motif_msa_list,
        "motif_gaps": motif_gaps,
        "motif_gaps_cluster": motif_gaps_cluster,
        "paste_locs": paste_locs,
        "functional_sites": functional_sites,
        "motif_target": motif_target,
        "target_pdbs": target_pdbs,
        "target_chains": target_chains,
        "contig_locs_list": contig_locs_list,
        "contig_locs_chain_list": contig_locs_chain_list,
    }


def read_motif(path, logger):
    if os.path.exists(path):
        data = load_pickle(path)
        motif_msa_list = data["motif_msa_list"]
        motif_atom14_coord = data["motif_atom14_coord_list"]
        motif_pdb_list = data["motif_pdb_list"]
        motif_with_pdb_index = data["motif_with_pdb_index"]
        motif_locs_list = data["motif_locs_list"]
        motif_locs_chain_list = data["motif_locs_chain_list"]
        motif_cluster = data["motif_cluster"]
        motif_template = data["motif_template"]
        special_motif_index = data["special_motif_index"]
        special_motif_expand = data["special_motif_expand"]
        expand_motif_loc_list = data["expand_motif_loc_list"]
        expand_motif_msa_list = data["expand_motif_msa_list"]
        motif_gaps = data["motif_gaps"]
        motif_gaps_cluster = data["motif_gaps_cluster"]
        paste_locs = data["paste_locs"]
        functional_sites = data["functional_sites"]
        motif_target = data["motif_target"]
        target_pdbs = data["target_pdbs"]
        target_chains = data["target_chains"]
        contig_locs_list = data["contig_locs_list"]
        contig_locs_chain_list = data["contig_locs_chain_list"]

        logger.debug("*" * 100)
        logger.debug(f"motif_pdb_list: {motif_pdb_list}")
        logger.debug(f"motif_with_pdb_index: {motif_with_pdb_index}")
        logger.debug(f"motif_locs_list: {motif_locs_list}")
        logger.debug(f"motif_locs_chain_list: {motif_locs_chain_list}")
        logger.debug(f"motif_cluster: {motif_cluster}")
        logger.debug(f"motif_template: {motif_template}")
        logger.debug(f"special_motif_index: {special_motif_index}")
        logger.debug(f"special_motif_expand: {special_motif_expand}")
        logger.debug(f"expand_motif_loc_list: {expand_motif_loc_list}")
        logger.debug(f"motif_gaps: {motif_gaps}")
        logger.debug(f"motif_gaps_cluster: {motif_gaps_cluster}")
        logger.debug(f"paste_locs: {paste_locs}")
        logger.debug(f"functional_sites: {functional_sites}")
        logger.debug(f"motif_target: {motif_target}")
        logger.debug(f"target_pdbs: {target_pdbs}")
        logger.debug(f"target_chains: {target_chains}")
        logger.debug(f"contig_locs_list: {contig_locs_list}")
        logger.debug(f"contig_locs_chain_list: {contig_locs_chain_list}")
        logger.debug("*" * 100)

        return (
            motif_msa_list,
            motif_atom14_coord,
            motif_pdb_list,
            motif_with_pdb_index,
            motif_locs_list,
            motif_locs_chain_list,
            motif_cluster,
            motif_template,
            special_motif_index,
            special_motif_expand,
            expand_motif_loc_list,
            expand_motif_msa_list,
            motif_gaps,
            motif_gaps_cluster,
            paste_locs,
            functional_sites,
            motif_target,
            target_pdbs,
            target_chains,
            contig_locs_list,
            contig_locs_chain_list,
        )
    else:
        return [None for i in range(21)]
