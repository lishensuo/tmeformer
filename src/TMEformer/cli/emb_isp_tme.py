#!/usr/bin/env python3
"""
Cell Embedding Similarity Perturbation Analysis Script 
- Predict the change of cell embedding similarity for TME-Rank/Composition ISP
"""

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    window: int = 0
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
    max_isp_cells: int


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
# ISP_TME_SIM_Analyzer Class
# ============================================================================


class ISP_TME_SIM_Analyzer:
    """Analyzer for TME cell embedding similarity perturbation analysis."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.proj = args.proj
        self.model_ids = args.model_ids
        self.task = args.task
        self.direction = args.direction

        self.tme_method = args.tme_method
        self.cell_cluster_file = args.cell_cluster_file
        self.isp_cluster = args.isp_cluster
        self.fixed_clusters = args.fixed_clusters
        self.isp_genes = args.gene_list
        self.gene_mode = args.gene_mode

        self.window = args.window
        self.endpoints = args.endpoints
        self.ko_method = args.ko_method
        self.ki_method = args.ki_method

        self.cell_ratio = args.cell_ratio
        self.intra_sample = args.intra_sample
        self.custom_log_suffix = args.custom_log_suffix
        self.do_force = args.do_force

        self.work_dir = args.work_dir

        self._set_project_config()
        self._setup_paths_and_configs()
        self._load_cell_cluster_dict()
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
        )

    def _setup_paths_and_configs(self):
        """Setup paths and configurations."""
        if self.cell_ratio is not None:
            self.proj_config.max_isp_cells = self.cell_ratio

        if self.tme_method == "rank":
            if self.gene_mode == "combination":
                self.isp_gene_lists = [sorted(self.isp_genes)]
            elif self.gene_mode == "single":
                self.isp_gene_lists = [[gene] for gene in self.isp_genes]

        elif self.tme_method == "composition":
            self.isp_gene_lists = [None]

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

    def _load_cell_cluster_dict(self):
        """Load cell cluster dictionary."""
        with open(Path(self.work_dir) / f"data/{self.proj}/processed/{self.proj}_tme_id_dict.pkl", "rb") as f:
            main_tme_id_dict = pickle.load(f)
        main_tme_ids = set(main_tme_id_dict.values())

        isp_proj = self.proj 
        cell_tme_file_path = Path(self.work_dir) / f"data/{isp_proj}/processed/{isp_proj}_cell_cluster_main_dict.pkl"
        with open(cell_tme_file_path, "rb") as f:
            self.cell_tme_dict = pickle.load(f)

        if (self.isp_cluster in main_tme_ids) and (self.cell_cluster_file == "main"):
            cell_cluster_file_path = cell_tme_file_path
        elif (self.isp_cluster not in main_tme_ids) and (self.cell_cluster_file != "main"):
            cell_cluster_file_path = Path(self.work_dir) / f"data/{isp_proj}/processed/{isp_proj}_cell_cluster_{self.cell_cluster_file}_dict.pkl"
        else:
            raise ValueError(f"Please check isp_cluster({self.isp_cluster}) and cell_cluster_file({self.cell_cluster_file}) are matched.")

        if cell_cluster_file_path.exists():
            with open(cell_cluster_file_path, "rb") as f:
                self.cell_cluster_dict = pickle.load(f)
        else:
            raise ValueError(f"{cell_cluster_file_path} not found.")

        # subcell_cluster_dict = {
        #     11: self.work_dir + "data/xenium/celltypes/P021_myCAF_cell_cluster_dict.pkl",
        #     12: self.work_dir + "data/xenium/celltypes/P021_myCAF_cell_cluster_dict.pkl",
        # }
    def _setup_logger(self):
        """Setup logging configuration."""
        config_suffix = tu_isp_ds.isp_run_suffix(self.isp_config.__dict__)

        self.tme_method_label = {
            "composition": "TME_Composition",
            "rank": "TME_Rank",
        }[self.tme_method]

        self.pre_save_name = (
            f"{self.proj}_{self.direction}-{self.tme_method_label}-"
            f"L{abs(self.model_config.embed_layer)}-"
            f"S{self.proj_config.max_isp_cells}-{config_suffix}"
        )
        self.task_dir = self.work_dir + f"isp_emb_sim/task_{self.task}"
        log_name = self._make_full_save_name(
            self.pre_save_name, self.tme_method, self.isp_cluster,
            self.fixed_clusters, self.isp_gene_lists
        )

        log_dir = Path(self.task_dir) / "log" / self.tme_method_label.lower()
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
                logging.Formatter("%(asctime)s [ISP TME SIM] %(message)s")
            )
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP TME SIM] %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _make_full_save_name(self, pre_save_name: str,
                             tme_method: str,
                             isp_cluster: int,
                             fixed_clusters: list,
                             isp_gene: Optional[List] = None) -> str:
        """Make full save name for output files."""
        if tme_method == "rank":
            if isinstance(isp_gene[0], list):
                isp_gene_text = "_".join(["+".join(gene) for gene in isp_gene])
            elif isinstance(isp_gene[0], str):
                isp_gene_text = "+".join(isp_gene)
            return f"{pre_save_name}-TME{isp_cluster}_{isp_gene_text}"
        elif tme_method == "composition":
            fixed_label = "all" if fixed_clusters is None else "+".join(str(i) for i in fixed_clusters)
            return f"{pre_save_name}-TME{isp_cluster}_FIX{fixed_label}"

    def _check_model_ids(self):
        """Check and filter valid model IDs."""
        all_model_ids = list(self.model_config.model_dict.keys())
        valid_model_ids = [
            model_id for model_id in self.model_ids if model_id in all_model_ids
        ]

        # Update valid model_ids
        if len(valid_model_ids) == 0:
            self.logger.info("All models have previously been done!")
            exit()
        self.model_ids = valid_model_ids

    def _create_and_init_pipeline(self, model_ids):
        """Create and initialize the TME ISP pipeline."""
        task_tme_ispipe = TME_ISPipe(
            proj=self.proj,
            task_dir=self.task_dir,
            sample_meta_groups=self.task_config.sample_meta_groups,
            model_config=self.model_config.__dict__,
            tme_datasets_dict=self.proj_config.tme_datasets_dict,
            sample2id_dict_file=self.proj_config.sample2id_dict_file,
            isp_cell_ratio=self.proj_config.max_isp_cells,
            isp_run_config=self.isp_config.__dict__,
            cell_tme_dict=self.cell_tme_dict,
            cell_cluster_dict=self.cell_cluster_dict,
            logger=self.logger
        )

        task_tme_ispipe.initiate_data(
            sample_max_cells=self.proj_config.max_sample_cells,
            model_ids=model_ids,
            split_file=self.task_config.intra_split_file,
            force=False
        )

        return task_tme_ispipe

    def _run_and_stat_pipeline(self, task_tme_ispipe, gene_lists):
        """Run TME ISP and statistics pipeline."""
        for gene_list in gene_lists:
            task_tme_ispipe.run_isp_tme(
                tme_method=self.tme_method,
                direction=self.task_config.direction,
                isp_cluster_id=self.isp_cluster,
                fixed_cluster_ids=self.fixed_clusters,
                gene_symbol=gene_list,
                force=self.do_force
            )

        interval_dict = {
            "OE": [[0, 1]],
            "KD": [[0, 1]],
            "KO": [[0, 1]],
            "KI": [None]
        }

        task_tme_ispipe.stat_isp_tme(
            tme_method=self.tme_method,
            direction=self.task_config.direction,
            interval_dict=interval_dict,
            force=self.do_force
        )

    def run(self):
        """Run the TME ISP analysis pipeline."""
        self.logger.info(f"# Task    : {self.task_config.__dict__}")
        self.logger.info(f"# Params  : {self.args.__dict__}")

        task_tme_ispipe = self._create_and_init_pipeline(self.model_ids)
        self._run_and_stat_pipeline(task_tme_ispipe, self.isp_gene_lists)


# ============================================================================
# Argument Parser
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="ISP_TME_SIM_Analyzer")

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--proj", type=str, default="xenium",
                        choices=["xenium", "pca165"])
    parser.add_argument("--model_ids", type=str, nargs="+", default=None,
                        help='e.g. ["GF_PR", "GF_D0528_06]')
    parser.add_argument("--task", type=str)
    parser.add_argument("--direction", type=str, default=None)

    # ISP params
    parser.add_argument("--tme_method", type=str, default="composition",
                        choices=["composition", "rank"],
                        help="composition: isp ratio of one tme type; rank: isp gene expression of one tme_type")
    parser.add_argument("--cell_cluster_file", type=str, default="main")
    parser.add_argument("--isp_cluster", type=int, default=None,
                        help="choose one tme type to isp")
    parser.add_argument("--fixed_clusters", type=int, nargs="+", default=None,
                        help="set ratio of some tme types to fix when tme composition isp")
    parser.add_argument("--gene_list", type=str, nargs="+", default=None,
                        help="isp gene when tme rank isp")
    parser.add_argument("--gene_mode", type=str, default="single",
                        choices=["single", "combination"])

    parser.add_argument("--window", type=float, default=0)
    parser.add_argument("--endpoints", type=float, default=1)
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

    analyzer = ISP_TME_SIM_Analyzer(args)
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
                device="1",
                proj="xenium",
                model_ids=["GF_D0527_01"],
                task="P009ADT2P009CRPC",
                direction="P009ADT>P009CRPC",

                tme_method="composition",
                cell_cluster_file="main",
                isp_cluster=5,
                gene_list=None,
                gene_mode="single",

                endpoints=2,
                window=0.1,

                cell_ratio=1000,
                intra_sample=False,
                custom_log_suffix=None,
                do_force=False,

                work_dir="/dataSSD7T/liss/work/scPCa/model/"
            )
            analyzer = ISP_TME_SIM_Analyzer(args)
            analyzer.run()
        else:
            # Script mode
            main()
    except NameError:
        # Script mode
        main()