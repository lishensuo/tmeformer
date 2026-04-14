#!/usr/bin/env python3
"""
Cell Embedding Similarity Perturbation Analysis Script 
- Predict the change of cell embedding similarity for Target-Rank ISP
"""

import argparse
import json
import logging
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from IPython import get_ipython

from TMEformer.tme import TmeModeling_utils as tu
from TMEformer.tme import TmeModeling_utils_isp_ds as tu_isp_ds
from TMEformer.tme.TmeModeling_utils import str2bool
from TMEformer.tme.TmeModeling_utils_isp_pipe import TME_ISPipe


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ISPConfig:
    """ISP configuration parameters."""
    window: int = 100
    endpoints: int = 1
    ko_method: Optional[str] = None
    ki_method: Optional[str] = None


@dataclass
class ProjectConfig:
    """Project configuration parameters."""
    name: str
    tme_datasets_dict: Dict[str, str]
    sample2id_dict_file: str
    max_sample_cells: int
    max_isp_cells: int  # target cells and total bg cells
    bg_max_cells_per_isp: int
    bg_isp_n_times: int


@dataclass
class TaskConfig:
    """Task configuration parameters."""
    proj: str
    task: str
    direction: str
    sample_meta_groups: Dict[str, List[str]]
    intra_split_file: str


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    device: int
    batch_size: int = 4
    embed_layer: int = -1
    model_dict: Dict[str, List[str]] = None


# ============================================================================
# ISP_SIM_Analyzer Class
# ============================================================================


class ISP_SIM_Analyzer:
    """Analyzer for cell embedding similarity perturbation analysis."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.proj = args.proj
        self.model_ids = args.model_ids
        self.task = args.task
        self.direction = args.direction

        self.isp_genes = args.gene_list
        self.gene_mode = args.gene_mode

        self.window = args.window
        self.endpoints = args.endpoints
        self.ko_method = args.ko_method
        self.ki_method = args.ki_method

        self.cell_ratio = args.cell_ratio
        self.background = args.background
        self.intra_sample = args.intra_sample
        self.custom_log_suffix = args.custom_log_suffix
        self.do_force = args.do_force

        self.work_dir = args.work_dir

        self._set_project_config()
        self._setup_paths_and_configs()
        self._setup_logger()
        self._check_model_ids()

    def _set_project_config(self):
        """Set project configuration."""
        self.proj_config = ProjectConfig(
            name=self.proj,
            tme_datasets_dict=tu_isp_ds.tme_dataset_versions(
                proj=self.proj, celldata="epi", work_dir=self.work_dir
            ),
            sample2id_dict_file=self.work_dir + f"data/{self.proj}/processed/{self.proj}_sample_id_dict.pkl",
            max_sample_cells=10000,  # for stat embedding
            max_isp_cells=5000,
            bg_max_cells_per_isp=50,
            bg_isp_n_times=100
        )

    def _setup_paths_and_configs(self):
        """Setup paths and configurations."""
        if self.cell_ratio is not None:
            self.proj_config.max_isp_cells = self.cell_ratio

        if self.background == 0:
            if self.gene_mode == "combination":
                self.isp_gene_lists = [sorted(self.isp_genes)]
            elif self.gene_mode == "single":
                self.isp_gene_lists = [[gene] for gene in self.isp_genes]
        else:
            self.gene_mode = "single" if self.background == 1 else "combination"
            total_bg_cells = self.proj_config.bg_isp_n_times * self.proj_config.bg_max_cells_per_isp
            if total_bg_cells != self.proj_config.max_isp_cells:
                raise ValueError("total_bg_cells != max_isp_cells. Please check the proj_config.")
            self.isp_gene_lists = self._generate_bg_random_gene(
                self.proj, self.background, self.proj_config.bg_isp_n_times,
                seed=42, work_dir=self.work_dir
            )

        self.isp_config = ISPConfig(
            self.window if self.gene_mode == "single" else 0,
            self.endpoints, self.ko_method, self.ki_method
        )

        self.task_config = self._make_task_config(
            self.proj, self.task, self.direction, self.intra_sample, self.work_dir
        )

        if self.proj == "xenium":
            model_dict = tu.generate_pr_models_dict("checkpoint_xe", self.work_dir)

        self.model_config = ModelConfig(
            device=self.device, model_dict=model_dict
        )

    @staticmethod
    def _generate_bg_random_gene(proj, n_genes_per_bg, n_times_of_bg, seed=42, work_dir=None):
        """Generate random gene lists for background analysis."""
        config_path = work_dir + "isp_emb_sim/config_random_genes.json"
        with open(config_path, "r") as f:
            random_genes_dict = json.load(f)

        k = f"{proj}_S{seed}_B{n_genes_per_bg}_N{n_times_of_bg}"

        if k not in random_genes_dict:
            genes_valid = tu.get_all_valid_genes(proj, work_dir=work_dir)
            random.seed(seed)
            gene_lists = [
                sorted(random.sample(genes_valid.tolist(), k=n_genes_per_bg))
                for _ in range(n_times_of_bg)
            ]
            random_genes_dict[k] = gene_lists
            with open(config_path, "w") as f:
                json.dump(random_genes_dict, f, indent=4)
        else:
            gene_lists = random_genes_dict[k]

        return gene_lists

    @staticmethod
    def _make_task_config(proj, task, direction, intra_sample=False, work_dir=None) -> TaskConfig:
        """Make task configuration."""
        config_group_file = work_dir + f"isp_emb_sim/config_{proj}_group.json"
        with open(config_group_file, "r") as f:
            config_group = json.load(f)
        if task not in config_group.keys():
            raise ValueError(f"{task} not in {config_group.keys()}")

        sample_meta_groups = config_group[task]
        valid_groups = [group for group in sample_meta_groups if group != "PID"]

        # Check direction
        if ">" not in direction:
            raise ValueError("direction must be in format of StateA>StateB")
        groups = direction.split(">")
        if not set(groups).issubset(valid_groups):
            raise ValueError(f"{groups} not all in {valid_groups}")

        if not intra_sample:
            intra_split_file = None
        else:
            intra_split_file = work_dir + f"isp_emb_sim/task_{task}/isp_split.csv"
            if not os.path.exists(intra_split_file):
                raise ValueError("intra_split_file not exists, please generate it first.")

        return TaskConfig(proj, task, direction, sample_meta_groups, intra_split_file)

    def _setup_logger(self):
        """Setup logging configuration."""
        config_suffix = tu_isp_ds.isp_run_suffix(self.isp_config.__dict__)
        if self.background == 0:
            self.pre_save_name = (
                f"{self.task}_{self.direction}-Target_Rank-"
                f"L{abs(self.model_config.embed_layer)}-"
                f"S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        else:
            self.pre_save_name = (
                f"{self.task}_{self.direction}-Target_Rank-"
                f"L{abs(self.model_config.embed_layer)}-"
                f"S{self.proj_config.bg_max_cells_per_isp}-{config_suffix}"
            )

        self.task_dir = self.work_dir + f"isp_emb_sim/task_{self.task}"
        log_name = self._make_full_save_name(self.pre_save_name, self.isp_gene_lists, self.background)
        log_dir = Path(self.task_dir) / "log" / "target_rank"
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.custom_log_suffix:
            log_name += "_" + self.custom_log_suffix
        logfile = log_dir / f"{log_name}.log"

        # Setup logger
        self.logger = logging.getLogger(f"{log_name}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(logfile, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP SIM] %(message)s")
            )
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP SIM] %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _make_full_save_name(self, pre_save_name: str,
                             isp_gene: Optional[List] = None,
                             background: int = 0) -> str:
        """Make full save name for output files."""
        if background == 0:
            if isinstance(isp_gene[0], list):
                isp_gene_text = '_'.join(['+'.join(gene) for gene in isp_gene])
            elif isinstance(isp_gene[0], str):
                isp_gene_text = '+'.join(isp_gene)
            return f"{pre_save_name}-{isp_gene_text}"
        elif background > 0:
            return f"{pre_save_name}-Background_C{background}"
        else:
            raise ValueError("Either isp_gene or background(>0) must be specified")

    def _check_model_ids(self):
        """Check and filter valid model IDs."""
        all_model_ids = list(self.model_config.model_dict.keys())
        valid_model_ids = [
            model_id for model_id in self.model_ids if model_id in all_model_ids
        ]

        # Check existed background results.
        # Because the actual operation is based on genes, not "Background_C1",
        # it cannot be checked during run.
        if self.background > 0:
            existed_model_ids = []
            for model_id in valid_model_ids:
                save_file = (
                    self._make_full_save_name(self.pre_save_name, background=self.background) + ".csv"
                )
                save_dir = Path(self.task_dir) / "stat" / model_id / "target_rank"
                if (save_dir / save_file).exists():
                    existed_model_ids.append(model_id)
                    self.logger.info(f"==> Skip: Background ISP for {model_id} has been done.")
            valid_model_ids = [
                model_id for model_id in valid_model_ids if model_id not in existed_model_ids
            ]

        # Update valid model_ids
        if len(valid_model_ids) == 0:
            self.logger.info("All models have been done!")
            exit()
        self.model_ids = valid_model_ids

    def _create_and_init_pipeline(self, model_ids):
        """Create and initialize the ISP pipeline."""
        task_ispipe = TME_ISPipe(
            proj=self.proj,
            task_dir=self.task_dir,
            sample_meta_groups=self.task_config.sample_meta_groups,
            model_config=self.model_config.__dict__,
            tme_datasets_dict=self.proj_config.tme_datasets_dict,
            sample2id_dict_file=self.proj_config.sample2id_dict_file,
            isp_cell_ratio=(
                self.proj_config.max_isp_cells if self.background == 0
                else self.proj_config.bg_max_cells_per_isp
            ),
            isp_run_config=self.isp_config.__dict__,
            logger=self.logger
        )

        task_ispipe.initiate_data(
            sample_max_cells=self.proj_config.max_sample_cells,
            model_ids=model_ids,
            split_file=self.task_config.intra_split_file,
            force=False
        )

        return task_ispipe

    def _run_and_stat_pipeline(self, task_ispipe, gene_lists):
        """Run ISP and statistics pipeline."""
        for gene_list in gene_lists:
            task_ispipe.run_isp(
                gene_symbol=gene_list,
                direction=self.task_config.direction,
                force=self.do_force
            )

        if self.gene_mode == "single":
            interval_dict = {
                "OE": [[0, 1], [0.5, 1], [0.75, 1], [0.9, 1]],
                "KD": [[0, 1], [0, 0.5], [0, 0.25], [0, 0.1]],
                "KO": [[0, 1], [0, 0.5], [0, 0.25], [0, 0.1]],
                "KI": [None]
            }
        elif self.gene_mode == "combination":
            interval_dict = {
                "OE": [[0, 1], [0.25, 1], [0.5, 1], [0.75, 1]],
                "KD": [[0, 1], [0, 0.75], [0, 0.5], [0, 0.25]],
                "KO": [[0, 1], [0, 0.75], [0, 0.5], [0, 0.25]],
                "KI": [None]
            }

        task_ispipe.stat_isp(
            direction=self.task_config.direction,
            interval_dict=interval_dict,
            force=self.do_force
        )

    def _merge_bg_isp_results(self, background, model_ids, gene_lists, remove=True):
        """Merge background ISP results."""
        self.logger.info("==> To merge background ISP results")

        gene_lists_collapsed = ["+".join(gene_list) for gene_list in gene_lists]
        bg_file_names = [
            self._make_full_save_name(self.pre_save_name, gene_list)
            for gene_list in gene_lists
        ]

        for model_id in model_ids:
            bg_output_fls = [
                Path(self.task_dir) / "output" / model_id / "target_rank" / f"{bg_file_name}.pickle"
                for bg_file_name in bg_file_names
            ]
            bg_stat_fls = [
                str(bg_output_fl).replace("output", "stat").replace(".pickle", ".csv")
                for bg_output_fl in bg_output_fls
            ]

            bg_output_dict = {}
            bg_stat_dict = {}
            for bg_output_fl, bg_stat_fl, gene_list in zip(
                bg_output_fls, bg_stat_fls, gene_lists_collapsed
            ):
                with open(bg_output_fl, "rb") as f:
                    bg_output_dict[gene_list] = pickle.load(f)

                bg_stat_cell_df = pd.read_csv(bg_stat_fl)
                bg_stat_cell_df["gene"] = gene_list
                bg_stat_dict[gene_list] = bg_stat_cell_df

            # self.proj_config.max_isp_cells
            pre_save_name_total_bg = re.sub(
                r"S\d+", f"S{self.proj_config.max_isp_cells}", self.pre_save_name
            )
            bg_save_name = self._make_full_save_name(pre_save_name_total_bg, background=background)
            bg_save_path = Path(self.task_dir) / "output" / model_id / "target_rank"

            with open(bg_save_path / f"{bg_save_name}.pickle", "wb") as f:
                pickle.dump(bg_output_dict, f)

            pd.concat(list(bg_stat_dict.values())).to_csv(
                Path(str(bg_save_path).replace("/output/", "/stat/")) / f"{bg_save_name}.csv",
                index=False
            )
            self.logger.info(f"==> Done: {bg_save_name}")
            if remove:
                for flist in [bg_output_fls, bg_stat_fls]:
                    for f in flist:
                        if os.path.exists(f):
                            os.remove(f)

    def run(self):
        """Run the ISP analysis pipeline."""
        self.logger.info(f"# Task    : {self.task_config.__dict__}")
        self.logger.info(f"# Params  : {self.args.__dict__}")

        task_ispipe = self._create_and_init_pipeline(self.model_ids)
        self._run_and_stat_pipeline(task_ispipe, self.isp_gene_lists)

        if self.background > 0:
            self._merge_bg_isp_results(
                self.background, self.model_ids, self.isp_gene_lists
            )


# ============================================================================
# Argument Parser
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="ISP_SIM_Analyzer")

    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--proj", type=str, default="xenium",
                        choices=["xenium", "pca165"])
    parser.add_argument("--model_ids", type=str, nargs="+", default=None,
                        help='e.g. ["GF_PR", "GF_D0528_06]')
    parser.add_argument("--task", type=str)
    parser.add_argument("--direction", type=str, default=None)

    # ISP params
    parser.add_argument("--gene_list", type=str, nargs="+", default=None,
                        help='e.g. ["PTEN","TP53","RB1"]')
    parser.add_argument("--gene_mode", type=str, default="single",
                        choices=["single", "combination"],
                        help='"single": isp single gene; "combination": isp combination of multi-genes')
    parser.add_argument("--background", type=int, default=0,
                        help='0 表示不计算背景分布  >0 表示计算背景分布')

    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--endpoints", type=int, default=1)
    parser.add_argument("--ko_method", type=str, default=None)
    parser.add_argument("--ki_method", type=str, default=None)

    parser.add_argument("--cell_ratio", type=int, default=None)
    parser.add_argument("--intra_sample", type=str2bool, default=False)
    parser.add_argument("--custom_log_suffix", type=str, default=None)
    parser.add_argument("--do_force", type=str2bool, default=False)

    parser.add_argument("--work_dir", type=str, default="/dataSSD7T/liss/work/scPCa/model/")

    return parser


# ============================================================================
# CLI Main Entry
# ============================================================================


def main():
    """CLI main entry function."""
    parser = create_parser()
    args = parser.parse_args()

    analyzer = ISP_SIM_Analyzer(args)
    analyzer.run()


# ============================================================================
# Test and Debug Entry
# ============================================================================


if __name__ == "__main__":
    # Check if running in interactive environment (e.g., Jupyter)
    try:
        if get_ipython() is not None:
            # Interactive mode - for testing
            print("Running in interactive mode with test parameters...")
            args = argparse.Namespace(
                device="0",
                proj="xenium",
                model_ids=["GF_D1120_01"],
                task="ADT2CRPC",
                direction="ADT>CRPC",

                gene_list=["NKX2-8"],
                gene_mode="single",
                background=1,
                window=30,
                endpoints=1,

                cell_ratio=1000,
                intra_sample=False,
                custom_log_suffix=None,
                do_force=False,

                work_dir="/dataSSD7T/liss/work/scPCa/model/"
            )
            analyzer = ISP_SIM_Analyzer(args)
            analyzer.run()
        else:
            # Script mode
            main()
    except NameError:
        # Script mode
        main()