__all__ = ["main"]

import copy
import logging
import os
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Dict

import torch
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hallucination.log_tool import setup_logger, get_logger
from hallucination.motif_reader import load_motif
from hallucination.utils import write_pickle

setup_logger()

_FunctionList = List[Callable[[Any], Any]]


class Hook:
    # TODO: Move out
    def __init__(self, on_startup: _FunctionList, on_done: _FunctionList):
        self.on_startup = on_startup
        self.on_done = on_done

    def startup(self, **kwargs):
        for fn in self.on_startup:
            if fn is None:
                continue
            fn(**kwargs)

    def done(self, **kwargs):
        for fn in self.on_done:
            if fn is None:
                continue
            fn(**kwargs)


class HallucinateSubprocess(Hook):
    def __init__(
        self,
        *,
        config_file_path,
        output_dir,
        motif_pk_path,
        inpaint: bool,
        mcmc_indel: bool,
        code: str,
        mutate_idx,
        sampler_type="random",
        AA_freq_file=None,
        predict_model="trfold",
        init_method="random",
        init_sequence: str = None,  # path to a fasta file, or None
        init_mode="both",
        ngpu: int = 1,
        num_worker: int = 1,
        on_startup: List[Callable[[Any], Any]] = None,
        on_done: List[Callable[[Any], Any]] = None,
    ):
        self.logger = get_logger("hallucination-subprocess")
        self.output_dir = output_dir
        if on_startup is None:
            on_startup = []
        if on_done is None:
            on_done = []
        Hook.__init__(self, on_startup, on_done)
        cmd = [
            "python",
            "hallucination/generate.py",
            "--output-dir",
            f"{self.output_dir}",
            "--motif-pk-path",
            f"{motif_pk_path}",
        ]
        if ngpu != 0:
            cmd.append("--gpu")

        if config_file_path is not None:
            cmd.append("--config")
            cmd.append(config_file_path)
        cmd.append("--code")
        cmd.append(code)
        cmd.append("--init-method")
        cmd.append(init_method)
        if init_method in ["external_sequence", "external_both"]:
            assert init_sequence is not None, "init_sequence must be fasta file path"
            cmd.append("--init-sequence")
            cmd.append(init_sequence)
        if sampler_type != "random":
            cmd.append("--sampler-type")
            cmd.append(sampler_type)
        if AA_freq_file is not None:
            cmd.append("--AA-freq-file")
            cmd.append(AA_freq_file)
        if predict_model != "trfold":
            cmd.append("--predict-model")
            cmd.append(predict_model)
        if mutate_idx is not None:
            cmd.append("--mutate-idx")
            cmd.append(mutate_idx)
        cmd.append("--init-mode")
        cmd.append(init_mode)
        if inpaint:
            cmd.append("--inpaint")
        if mcmc_indel:
            cmd.append("--mcmc-indel")
        self.cmd = cmd
        self.ngpu: int = min(ngpu, torch.cuda.device_count())
        assert self.ngpu > 0, "ngpu must be greater than 0"
        self.thread_pools: List[ThreadPoolExecutor] = [
            ThreadPoolExecutor(
                max_workers=num_worker, thread_name_prefix=f"generator-pool-{i}"
            )
            for i in range(self.ngpu)
        ]  # each gpu has a thread pool
        self.future_list: List[Future] = []
        self.pool_idx = 0

        self.done_counter = 0

        def counter_inc(**kwargs):
            self.done_counter += 1

        def show_cmdline(**kwargs):
            self.logger.info(kwargs.get("cmdline", "NO_CMDLINE_SET"))

        def print_all_args(**kwargs):
            for k, v in kwargs.items():
                self.logger.info(f"{k} : {v}")

        self.on_startup.append(print_all_args)
        self.on_done.append(counter_inc)
        self.on_done.append(show_cmdline)

    @staticmethod
    def add_locs(cmd: List[str], cmd_prefix, locs: List[Tuple[int, int]]):
        for loc in locs:
            l, r = loc
            cmd.append(cmd_prefix)
            cmd.append(f"{l}-{r}")

    def run(
        self,
        prefix: str,
        trial_id: int,
        length_tuple,
    ):
        cmd = copy.copy(self.cmd)
        cmd.extend(
            [
                "--prefix",
                f"{prefix}",
                "--length-tuple",
                f"{length_tuple}",
                "--trial-id",
                f"{trial_id}",
            ]
        )

        process_env = {
            "HOME": "/root",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.environ.get("PYTHONPATH", "/app:/hallucination"),
            "PATH": os.environ.get("PATH"),
            "LD_LIBRARY_PATH": os.environ.get(
                "LD_LIBRARY_PATH",
                "/root/miniconda3/envs/xlab/lib/python3.9/site-packages/torch/lib",
            ),
            "CUDA_VISIBLE_DEVICES": str(self.pool_idx),
            "TZ": "Asia/Shanghai",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "False",
        }

        def thread_core(cmdline, envdict):
            self.startup(
                cmdline=cmdline,
                prefix=prefix,
                length_tuple=length_tuple,
                trial_id=trial_id,
            )
            with open(
                os.path.join(self.output_dir, f"generate-{trial_id}.log"), "a+"
            ) as logfile:
                subprocess.check_call(
                    cmdline,
                    stdout=logfile,
                    stderr=logfile,
                    cwd=os.path.realpath(os.path.join(os.path.dirname(__file__), "..")),
                    env=envdict,
                )
                self.done(cmdline=cmdline, count=self.done_counter)

        self.future_list.append(
            self.thread_pools[self.pool_idx].submit(thread_core, cmd, process_env)
        )
        self.pool_idx = (self.pool_idx + 1) % self.ngpu

    def join(self):
        try:
            self.logger.info("joining thread pool")
            for f in self.future_list:
                f.result()
        except Exception as e:
            raise e
        finally:
            for pool in self.thread_pools:
                pool.shutdown(wait=False)


def main(
    *,
    length_range: Tuple[str],
    motif_locs_list: Optional[List[Tuple[int, int]]] = None,
    motif_locs_chain_list: Optional[List[str]] = None,
    motif_cluster: Optional[str] = None,
    motif_template: Optional[str] = None,
    motif_with_pdb_index: Optional[List[int]] = None,
    special_motif_index: Optional[List[int]] = None,
    special_motif_extra_seqs: Optional[List[List[str]]] = None,
    special_motif_expand: Optional[List[List[int]]] = None,
    motif_gaps: Optional[List[List[int]]] = None,
    motif_gaps_cluster: Optional[Dict] = None,
    motif_pdb_list: Optional[List[str]] = None,
    prefix: str,  # task name
    trials: int,  # number of outputs
    init_trial: int,  # number of start idx, set to 0 if no need
    config_file_path: str,
    paste_locs: Optional[List[Tuple[int, int]]] = None,
    functional_sites: Optional[List[str]] = None,
    output_dir: str,
    inpaint: bool,
    mcmc_indel: bool,
    mutate_idx,
    sampler_type: str,  # one of random or plddt
    AA_freq_file: str,
    predict_model: str,  # predict_model
    init_method: str,  # one of random or external_sequence or external_template
    init_sequence,
    init_mode: str,  # one of before_gd or after_gd
    code: str,
    num_worker: int = 3,  # number of processes on one device
    ngpu: int = 10,
    motif_target: List[bool] = None,
    target_pdbs: List[str] = None,
    target_chains: Optional[List[List[str]]] = None,
    contig_locs_list: List[Tuple[int, int]],
    contig_locs_chain_list: List[str],
    on_trial_done: Optional[Callable[[Any], None]] = None,
):
    logger = logging.getLogger("hallucination")
    if len(motif_locs_list) > 0:
        _path_motif = Path(motif_pdb_list[0])
        motif_pk_path = _path_motif.with_suffix(".pk")
        motif_pk_path = motif_pk_path.absolute()
        if not motif_pk_path.exists():
            write_pickle(
                load_motif(
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
                ),
                str(motif_pk_path),
            )
        motif_pk_path = str(motif_pk_path)

    else:
        motif_pk_path = None

    hallucination_subprocess = HallucinateSubprocess(
        config_file_path=config_file_path,
        output_dir=output_dir,
        motif_pk_path=motif_pk_path,
        inpaint=inpaint,
        mcmc_indel=mcmc_indel,
        mutate_idx=mutate_idx,
        sampler_type=sampler_type,
        AA_freq_file=AA_freq_file,
        predict_model=predict_model,
        init_method=init_method,
        init_sequence=init_sequence,
        init_mode=init_mode,
        code=code,
        ngpu=ngpu,  # use all gpu. set to device_count inside
        num_worker=num_worker,  # number of worker on one gpu
        on_done=[on_trial_done] if on_trial_done else None,
    )

    try:
        trial_idx = 0
        max_try_time = 10000  # enough try time
        for try_time in range(max_try_time):
            if trial_idx >= trials:
                break
            logger.info(
                f"start trial No.{try_time + 1}/{max_try_time} time, trial_idx is {trial_idx + init_trial}"
            )
            length_temp = []
            for unit_length in length_range:
                length_split = unit_length.split("-")
                if init_method in ["random", "external_template"]:
                    if len(length_split) >= 2:
                        assert int(length_split[0]) < int(length_split[1])
                        length_temp.append(
                            random.randint(int(length_split[0]), int(length_split[1]))
                        )
                    elif len(length_split) == 1:
                        length_temp.append(int(length_split[0]))
                else:
                    assert len(length_split) == 1
                    length_temp.append(int(length_split[0]))
            length_tuple = tuple(length_temp)

            hallucination_subprocess.run(
                prefix=prefix,
                trial_id=trial_idx + init_trial,
                length_tuple=length_tuple,
            )
            trial_idx += 1
        hallucination_subprocess.join()
    except KeyboardInterrupt:
        for p in hallucination_subprocess.thread_pools:
            p.shutdown(wait=False)
    return
