from yacs.config import CfgNode

_C = CfgNode()
_C.steps = 5
_C.track_step = 1
_C.mcmc_steps = 5

## trfold config
_C.trfold = CfgNode()
_C.trfold.ckpt_path = "./params/trfold_params/ckpt_epoch_158"
_C.trfold.recycles = 1
_C.trfold.bf16 = True
# opt config
_C.opt = CfgNode()
_C.opt.optimizer = "nsgd"
_C.opt.learning_rate = 5e-2

## alphafold config
_C.alphafold = CfgNode()
_C.alphafold.data_dir = "./params/alphafold_params/"
_C.alphafold.model_id = 4
_C.alphafold.recycles = 1
_C.alphafold.msa_clusters = 1


# loss config
_C.loss = CfgNode()
_C.loss.rmsd_thresh = 0.0
_C.loss.earlystop_rmsd_thresh = 0.5
_C.loss.earlystop_steps = 1000
_C.loss.rog_thresh = 18.0
_C.loss.w_cce = 0.5
_C.loss.w_entropy = 0.1
_C.loss.w_fape = 1.0
_C.loss.charge_thresh = -4
_C.loss.w_nc = -1.0
_C.loss.w_plddt = 1.0
_C.loss.w_ptm = 1.0
_C.loss.w_pae = 0.05
_C.loss.w_rmsd = 0.5
_C.loss.rep_sigma = 3.5
_C.loss.w_rep = 1.0
_C.loss.atr_sigma = 5.0
_C.loss.w_atr = -1.0
_C.loss.w_rog = 10.0
_C.loss.w_surfnp = -1.0
_C.loss.w_motif_surf = -1.0
_C.loss.w_symmetry = 2.0
_C.loss.symmetry_type = "plane"  ##  select "plane" or "space"

# mcmc config
_C.mcmc = CfgNode()
_C.mcmc.mcmc_halflife = 500
_C.mcmc.earlystop = 500
_C.mcmc.T_acc_0 = 0.02
_C.mcmc.mcmc_batch = 1
_C.mcmc.num_mutated_tokens = 1
_C.mcmc.indel_prob = 0.5


def get_default_config() -> CfgNode:
    """
    Get a copy of the default config.

    Returns
        a CfgNode instance.
    """
    return _C.clone()
