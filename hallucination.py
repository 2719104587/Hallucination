import logging
import os
import click

from hallucination import main, setup_logger
from hallucination import preprocess_args as pa


@click.command("hallucination")
@click.option(
    "--length",
    type=str,
    required=True,
    help="eg: 80-120,90-130  Design protein length.",
)
@click.option(
    "--motif-locs",
    type=str,
    default=None,
    help="eg: A62-65,A93-97,B118-120  Location of structurally conservative areas.",
)
@click.option(
    "--motif-cluster",
    type=str,
    default=None,
    help="eg: A,A,B  Cluster of structural motifs, Motifs of the same Cluster remain spatially unchanged relative to each other. Priority takes precedence over relevance.",
)
@click.option(
    "--motif-template",
    type=str,
    default=None,
    help="eg: T,T,F  Template of structural motifs, T indicates that additional template information is required, and F indicates that no additional template information is required",
)
@click.option(
    "--motif-truth",
    type=str,
    default=None,
    help="eg: 1,1,2, The prefix location of each motif.",
)
@click.option(
    "--special-motif-index",
    type=str,
    default=None,
    help="eg: 1,2(functional-site not in motif) Additional conserved amino acids are required before and after motif",
)
@click.option(
    "--special-motif-extra-seqs",
    type=str,
    default=None,
    multiple=True,
    help="eg: LPDPSKPS,NKVTLAD For motif extension, comma before and after are the pre-extension part and the post-extension part respectively, This parameter is entered once for each special motif.",
)
@click.option(
    "--motif-gaps",
    type=str,
    default=None,
    help="eg: 0,10,10-20,0 Another way to specify the length of a design sequence, gap refers to the gap between motifs.",
)
@click.option(
    "--motif-gaps-cluster",
    type=str,
    default=None,
    help="eg: A,B,B,A Gaps in the same cluster will remain the same in length and amino acid type.",
)
@click.option(
    "--paste-locs",
    default=None,
    type=str,
    help="eg: 80-90,90-100 Manually specify the position of the motif on the sequence.",
)
## The number of motif intervals to match,
@click.option(
    "--motif-target",
    type=str,
    default=None,
    help="eg: T,T,F Whether motif needs to be designed with the help of additional target chain, T does, F does not",
)
@click.option(
    "--target-pdbs",
    type=str,
    default=None,
    help="eg: pdb_path1,pdb_path2,pdb_path3, pdb_path of the binding target.",
)
@click.option(
    "--target-chains",
    type=str,
    default=None,
    help="eg: HL,HL,HL Chain_id of the binding target.",
)
@click.option(
    "--combine-positions",
    type=str,
    default=None,
    multiple=True,
    help="eg: A62-65,A93-97,B118-120  The position in the motif interval that interacts with the target chain.",
)
@click.option(
    "--code",
    type=str,
    default=None,
    required=True,
    help="eg: ((A)(A)) Programming generate codex, Each code represents a chain, the chain sequence of the same code is consistent, and the code is the same and located at the same level represents symmetry.",
)
@click.option(
    "--motif-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option("--prefix", type=str, required=True, help="eg: 6xr8 ")
@click.option(
    "--trials",
    default=100,
    type=int,
    required=True,
    help="Desired design protein quantity for this round.",
)
@click.option(
    "--init-trial",
    default=1,
    type=int,
    required=True,
    help="The number of initial designs.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Configuration file path",
)
@click.option(
    "--functional-sites",
    type=str,
    default=None,
    help="eg: A62-65,A93 The position of the functional site. The functional site amino acid type remains unchanged and must be within the motif interval.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    required=True,
    help="Output directory path",
)
@click.option(
    "--inpaint",
    type=bool,
    is_flag=True,
    default=False,
    help="This feature is temporarily on hold",
)
@click.option(
    "--relevance",
    type=bool,
    is_flag=True,
    default=False,
    help="All motifs remain unchanged relative to each other in space.",
)
@click.option(
    "--mcmc-indel",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to use the MCMC algorithm to insert and delete residues.",
)
@click.option(
    "--sampler-type",
    type=click.Choice(["random", "plddt"], case_sensitive=False),
    default="random",
    help="random: randomly select the sequence, plddt: select the sequence with the lowest plddt.",
)
@click.option(
    "--aa-freq-file",
    type=str,
    default="/disk/dzk/hallucination_position/hallucination/config/AA_freq.json",
    help="Adjust the probability of each amino acid during mutation",
)
@click.option(
    "--predict-model",
    type=click.Choice(["trfold", "alphafold"], case_sensitive=False),
    default="trfold",
    help="trfold: trfold, alphafold: alphafold",
)
@click.option(
    "--mutate-idx",
    default=None,
    type=str,
    help="eg: 10-20,40-50  Manually specify which part of the amino acid sequence index can be mutated",
)
@click.option(
    "--initialization",
    type=click.Choice(
        ["random", "external_sequence", "external_template", "external_both"],
        case_sensitive=False,
    ),
    default="random",
    help="random: randomly generate the sequence, external_sequence: use the sequence in the fasta file, external_template: use the template in the pdb file, external_both: use the sequence and template in the fasta and pdb file",
)
@click.option(
    "--init-sequence",
    required=False,
    type=str,
    default=None,
    help="fasta_path, required if initialization is external_sequence or external_both",
)
@click.option(
    "--init-mode",
    type=click.Choice(["both", "gd", "mcmc"], case_sensitive=False),
    default="both",
    help="both: use both gd and mcmc, gd: use gd, mcmc: use mcmc",
)
@click.option(
    "--silence",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to print the log.",
)
@click.option("--ngpu", type=int, default=1, help="Number of GPU")
@click.option("--batchsize", type=int, default=3, help="Number of processes on one GPU")
def __click_main(
    length,
    motif_locs,
    motif_cluster,
    motif_template,
    motif_truth,
    special_motif_index,
    special_motif_extra_seqs,
    motif_gaps,
    motif_gaps_cluster,
    motif_target,
    target_pdbs,
    target_chains,
    combine_positions,
    code,
    motif_dir,
    prefix,
    trials,
    paste_locs,
    init_trial,
    config_file,
    functional_sites,
    output_dir,
    mutate_idx,
    sampler_type,
    aa_freq_file,
    predict_model,
    initialization,
    init_sequence,
    init_mode,
    inpaint,
    relevance,
    mcmc_indel,
    silence,
    ngpu,
    batchsize,
):
    (
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
        motif_pdb_list,
    ) = pa.pp_motif_args(
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
    )

    (
        motif_target,
        target_pdbs,
        target_chains,
        contig_locs_chain_list,
        contig_locs_list,
    ) = pa.pp_target_args(motif_target, target_pdbs, target_chains, combine_positions)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if silence:
        logger = logging.getLogger("hallucination")
        logger.setLevel(logging.NOTSET)

    length = tuple(length.split(","))

    prefix = prefix.split(",")[0]

    main(
        length_range=length,
        motif_locs_list=motif_locs_list,
        motif_locs_chain_list=motif_locs_chain_list,
        motif_cluster=motif_cluster,
        motif_template=motif_template,
        motif_with_pdb_index=motif_with_pdb_index,
        special_motif_index=special_motif_index,
        special_motif_extra_seqs=special_motif_extra_seqs,
        special_motif_expand=special_motif_expand,
        motif_gaps=motif_gaps,
        motif_gaps_cluster=motif_gaps_cluster,
        motif_target=motif_target,
        target_pdbs=target_pdbs,
        target_chains=target_chains,
        contig_locs_list=contig_locs_list,
        contig_locs_chain_list=contig_locs_chain_list,
        code=code,
        motif_pdb_list=motif_pdb_list,
        prefix=prefix,
        trials=trials,
        init_trial=init_trial,
        config_file_path=config_file,
        paste_locs=paste_locs,
        functional_sites=functional_sites,
        output_dir=output_dir,
        inpaint=inpaint,
        mcmc_indel=mcmc_indel,
        mutate_idx=mutate_idx,
        sampler_type=sampler_type,
        AA_freq_file=aa_freq_file,
        predict_model=predict_model,
        init_method=initialization,
        init_sequence=init_sequence,
        init_mode=init_mode,
        ngpu=ngpu,
        num_worker=batchsize,
    )


if __name__ == "__main__":
    setup_logger()
    __click_main()
