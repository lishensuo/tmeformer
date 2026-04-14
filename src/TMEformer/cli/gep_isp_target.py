#!/usr/bin/env python3
"""
Gene Expression Perturbation Analysis Script 
- Predict the change of marker gene expression for Target-Rank ISP
"""

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_from_disk

import TMEformer
from TMEformer import perturber_utils as pu
from TMEformer.tme import TmeModeling_utils as tu
from TMEformer.tme import TmeModeling_utils_isp_ds as tu_isp_ds
from TMEformer.tme import TmeModeling_utils_isp_gep as tu_isp_gep
from TMEformer.tme.TmeModeling_utils import str2bool


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
    max_isp_cells: Dict[str, int]
    bg_max_cells_per_isp: int
    bg_isp_n_times: int


# ============================================================================
# Project Configuration Constants
# ============================================================================


PROJECT_CONFIGS = {
    "xenium": ProjectConfig(
        name="xenium",
        max_isp_cells=10000,
        bg_max_cells_per_isp=100,
        bg_isp_n_times=100
    ),
    "xenium_10X_bca": ProjectConfig(
        name="xenium_10X_bca",
        max_isp_cells=10000,
        bg_max_cells_per_isp=100,
        bg_isp_n_times=100
    ),
    "xenium_10X_bca2": ProjectConfig(
        name="xenium_10X_bca2",
        max_isp_cells=10000,
        bg_max_cells_per_isp=100,
        bg_isp_n_times=100
    ),
    "xenium_10X_cc": ProjectConfig(
        name="xenium_10X_cc",
        max_isp_cells=10000,
        bg_max_cells_per_isp=100,
        bg_isp_n_times=100
    ),
    "xenium_10X_ov": ProjectConfig(
        name="xenium_10X_ov",
        max_isp_cells=10000,
        bg_max_cells_per_isp=100,
        bg_isp_n_times=100
    )
}


# ============================================================================
# ISP_GEP_Analyzer Class
# ============================================================================


class ISP_GEP_Analyzer:
    """Analyzer for gene expression perturbation analysis."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.proj = args.proj
        self.model_ids = args.model_ids
        self.marker_set = args.marker_set
        self.marker_id = args.marker_id
        self.isp_genes = args.gene_list
        self.gene_mode = args.gene_mode

        self.window = args.window
        self.endpoints = args.endpoints
        self.ko_method = args.ko_method
        self.ki_method = args.ki_method

        self.background = args.background
        self.cell_ratio = args.cell_ratio
        self.do_force = args.do_force
        self.pred_cells = args.pred_cells
        self.custom_log_suffix = args.custom_log_suffix

        self.work_dir = args.work_dir

        self.proj_config = PROJECT_CONFIGS[self.proj]
        
        with open(self.work_dir + "isp_gene_exp/MODEL_FT_GENE_SET.json", "r") as f:
            FT_Multi_SETs = json.load(f)
        self.model_genes = FT_Multi_SETs[self.marker_set]

        self._setup_paths_and_configs()

    def _setup_paths_and_configs(self):
        """Setup paths and configurations."""
        if self.pred_cells == "internal":
            self.file_suffix = ""
        else:
            self.file_suffix = "_" + self.pred_cells

        isp_window = self.window if (
            self.gene_mode == "single" and self.background == 0
        ) or (self.background == 1) else 0
        self.isp_config = ISPConfig(
            window=isp_window, endpoints=self.endpoints,
            ki_method=self.ki_method, ko_method=self.ko_method
        )

        if self.background == 0:
            if self.gene_mode == "combination":
                self.isp_gene_list = [sorted(self.isp_genes)]
            elif self.gene_mode == "single":
                self.isp_gene_list = [[gene] for gene in self.isp_genes]
        else:
            self.isp_gene_list = None

        if self.cell_ratio is not None:
            self.proj_config.max_isp_cells = self.cell_ratio

        if self.background > 0:
            total_bg_cells = (
                self.proj_config.bg_isp_n_times * self.proj_config.bg_max_cells_per_isp
            )
            if total_bg_cells != self.proj_config.max_isp_cells:
                raise ValueError(
                    "total_bg_cells != max_isp_cells. Please check the proj_config."
                )
            self._load_background_data()

    def _load_background_data(self):
        """Load background data for analysis."""
        # For ki, use different cells for isp
        ki_label = "_ki" if self.isp_config.ki_method is not None else ""
        bg_path = (
            Path(self.work_dir) / "isp_gene_exp" / self.proj / "datasets" / "background" /
            f"random_{self.background}genes_bg_cells_Model_{self.marker_set}{self.file_suffix}{ki_label}.dict"
        )

        if not bg_path.exists():
            raise FileNotFoundError(
                f"{bg_path} not exist. \n Please use tu_isp_gep.prep_bg_gene_lists() to prepare it."
            )

        with open(bg_path, "rb") as f:
            self.random_genes_bg_cells_dict = pickle.load(f)

    def _load_model_config(self):
        """Load model configuration."""
        config_path = Path(self.work_dir) / "isp_gene_exp/MODEL_FT_ID_MAP.json"
        with open(config_path, "r") as f:
            model_id_json = json.load(f)

        try:
            model_id_dict = model_id_json[self.proj][self.marker_set]
        except KeyError:
            available_keys = list(model_id_json[self.proj].keys())
            raise KeyError(f"Valid marker set:  {available_keys}")

        # Reverse dict {"FT_MODEL_ID": "GF_MODEL_ID"} -> {"GF_MODEL_ID": "FT_MODEL_ID"}
        self.model_id_dict = {v: k for k, v in model_id_dict.items()}

        # Filter specified models
        if self.model_ids is not None:
            for model_id in self.model_ids:
                if model_id not in self.model_id_dict:
                    raise ValueError(f"Invalid model id: {model_id}")
            self.model_id_dict = {
                k: v for k, v in self.model_id_dict.items()
                if k in self.model_ids
            }
        else:
            self.model_id_dict = model_id_dict

    def _setup_logger(self):
        """Setup logging configuration."""
        config_suffix = tu_isp_ds.isp_run_suffix(self.isp_config.__dict__)
        if self.marker_id is None:
            self.pre_save_name = (
                f"{self.marker_set}{self.file_suffix}-Target_Rank-S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        else:
            self.pre_save_name = (
                f"{self.marker_set}_{self.marker_id}{self.file_suffix}-Target_Rank-S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        log_name = self._make_full_save_name(
            self.pre_save_name, self.isp_gene_list, self.background
        )

        log_dir = (
            Path(self.work_dir) / "isp_gene_exp" / self.proj / "log_isp" /
            self.marker_set / "target_rank"
        )
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
                logging.Formatter("%(asctime)s [ISP GEP] %(message)s")
            )
            self.logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP GEP] %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _make_full_save_name(self, pre_save_name: str,
                             isp_gene: Optional[List] = None,
                             background: int = 0) -> str:
        """Make full save name for output files."""
        if background == 0:
            if isinstance(isp_gene[0], list):
                if len(isp_gene) > 5:
                    isp_gene_text = "_".join(
                        ["+".join(gene) for gene in isp_gene[:5]]
                    )
                    isp_gene_text += f"_and_{len(isp_gene)-5}more"
                else:
                    isp_gene_text = "_".join(["+".join(gene) for gene in isp_gene])
            elif isinstance(isp_gene[0], str):
                isp_gene_text = "+".join(isp_gene)
            return f"{pre_save_name}-{isp_gene_text}"
        elif background > 0:
            return f"{pre_save_name}-Background_C{background}"
        else:
            raise ValueError("Either isp_gene or background(>0) must be specified")

    def _load_model_and_dataset(self, model_id: str, ft_model_id: str):
        """Load model and dataset."""
        model_path = (
            Path(self.work_dir) / "isp_gene_exp" / self.proj / "output_ft" /
            self.marker_set / ft_model_id / "train_all" / "geneformer_cellRegressor_all"
        )

        model = pu.load_model(
            "CellClassifier_TME_MultiEXP",
            num_classes=len(self.model_genes),
            mode="eval",
            model_directory=str(model_path),
            quantize=False,
            device=f"cuda:{self.device}"
        )

        # Validate xenium model on external data
        if hasattr(model.config, "tme_config") and self.pred_cells != "internal":
            model.bert.cell_bert.tme_config.gf_tme_emb_path = (
                f"data/{self.pred_cells}/gf_emb/GF_CL_L0/"
            )

        # Load dataset
        PR_XE_MODELS_DICT = tu.generate_pr_models_dict("checkpoint_xe", self.work_dir)
        version = PR_XE_MODELS_DICT[model_id][1]
        dataset_path = (
            Path(self.work_dir) / "isp_gene_exp" / self.proj / "datasets" /
            f"isp_{version}_{self.marker_set}{self.file_suffix}.dataset"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"{dataset_path} not exist. \n Please prepare it."
            )

        dataset = load_from_disk(str(dataset_path))
        dataset = tu.modify_tme_dataset(model, dataset)

        return model, dataset

    def _process_isp(self, model, dataset, model_id: str, save_dir: Path):
        """Process ISP analysis."""
        for j, isp_gene in enumerate(self.isp_gene_list):
            save_path = save_dir / (
                self._make_full_save_name(self.pre_save_name, isp_gene) + ".csv"
            )

            self.logger.info(
                f"## Start model {model_id} | ISP {isp_gene} ({j+1}/{len(self.isp_gene_list)})"
            )
            self.logger.info(f"==> Save PATH: {save_path}")

            if not self.do_force and save_path.exists():
                self.logger.info("==> Skip: Found existing result file")
                continue

            score_df_list = []
            isp_gene_token = [tu.symbol2token(gene) for gene in isp_gene]

            for idx, model_gene in enumerate(self.model_genes):
                # 当指定预测的model gene时, 不处理set里的其它model gene
                if self.marker_id is not None and idx != self.marker_id:
                    continue
                dataset_filtered = tu_isp_ds.filter_token_dataset(
                    dataset, tokens=isp_gene_token,
                    existed=True if self.isp_config.ki_method is None else False,
                    ratio=self.proj_config.max_isp_cells
                )

                self.logger.info(
                    f"==> ISP cells: {len(dataset_filtered)} / {len(dataset)} "
                    f"for {model_gene}"
                )

                score_df = tu_isp_gep.calc_gep_cell_isp_score_from_cells(
                    dataset_filtered, isp_gene_token, model,
                    self.isp_config.__dict__, batch=8, pred_idx=idx, logger=self.logger
                )
                score_df["model_gene"] = model_gene
                score_df_list.append(score_df)

            cells_isp_score_df = pd.concat(score_df_list)
            cells_isp_score_df.to_csv(save_path)

    def _process_background_isp(self, model, dataset, model_id: str, save_dir: Path):
        """Process background ISP analysis."""
        config_suffix = tu_isp_ds.isp_run_suffix(self.isp_config.__dict__)
        if self.marker_id is None:
            pre_save_name = (
                f"{self.marker_set}{self.file_suffix}-Target_Rank-S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        else:
            pre_save_name = (
                f"{self.marker_set}_{self.marker_id}{self.file_suffix}-Target_Rank-S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        save_path = save_dir / (
            self._make_full_save_name(pre_save_name, background=self.background) + ".csv"
        )

        self.logger.info(f"==> Save PATH: {save_path}")

        if not self.do_force and save_path.exists():
            self.logger.info("==> Skip: Found existing result file")
            return

        bg_score_df_list = []
        items = list(self.random_genes_bg_cells_dict.items())

        for j, (isp_gene_list, valid_cell_id) in enumerate(items):
            self.logger.info(
                f"## Start model {model_id} | ISP {isp_gene_list} ({j+1}/{len(items)})"
            )

            isp_gene_token = [tu.symbol2token(gene) for gene in isp_gene_list]
            dataset_bg = dataset.select(valid_cell_id)

            score_df_list = []
            for idx, model_gene in enumerate(self.model_genes):
                # 当指定预测的model gene时, 不处理set里的其它model gene
                if self.marker_id is not None and idx != self.marker_id:
                    continue
                dataset_filtered = tu_isp_ds.filter_token_dataset(
                    dataset_bg, tokens=isp_gene_token,
                    existed=True if self.isp_config.ki_method is None else False,
                    ratio=self.proj_config.bg_max_cells_per_isp
                )

                score_df = tu_isp_gep.calc_gep_cell_isp_score_from_cells(
                    dataset_filtered, isp_gene_token, model,
                    self.isp_config.__dict__, pred_idx=idx, logger=self.logger
                )
                score_df["model_gene"] = model_gene
                score_df_list.append(score_df)

            cells_isp_score_df = pd.concat(score_df_list)
            cells_isp_score_df["isp_gene"] = "+".join(isp_gene_list)
            bg_score_df_list.append(cells_isp_score_df)

            if len(bg_score_df_list) == self.proj_config.bg_isp_n_times:
                cells_isp_score_bg_merge = pd.concat(
                    bg_score_df_list
                ).reset_index(drop=True)
                cells_isp_score_bg_merge.to_csv(save_path)
                break

    def run(self):
        """Run the GEP ISP analysis pipeline."""
        self._load_model_config()
        self._setup_logger()

        self.logger.info(f"# TASK   : {self.logger.name}")
        self.logger.info(f"# Params : {self.args.__dict__}")
        self.logger.info(f"# Models : {self.model_id_dict}")


        for i, (model_id, ft_model_id) in enumerate(self.model_id_dict.items()):
            self.logger.info(
                f"## Start model {model_id} ({i+1}/{len(self.model_id_dict)})"
            )

            # Load model and dataset
            model, dataset = self._load_model_and_dataset(model_id, ft_model_id)

            # Setup save directory
            save_dir = (
                Path(self.work_dir) / "isp_gene_exp" / self.proj / "output_isp" /
                model_id / self.marker_set / "target_rank"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            # Process based on analysis type
            if self.background == 0:
                self._process_isp(model, dataset, model_id, save_dir)
            else:
                self._process_background_isp(model, dataset, model_id, save_dir)


# ============================================================================
# Argument Parser
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="ISP_GEP_Analyzer")

    # Basic params
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--proj", type=str, default="xenium",
                        choices=["xenium", "xenium_10X_bca", "xenium_10X_bca2","xenium_10X_cc", "xenium_10X_ov"])
    parser.add_argument("--model_ids", type=str, nargs="+", default=None,
                        help='e.g. ["GF_PR", "GF_CL]')
    parser.add_argument("--marker_set", type=str, default=None,
                        help='e.g. "SET0"')
    parser.add_argument("--marker_id", type=int, default=None,
                        help='None for all markers or index id for one marker in the set'), 

    # ISP params
    parser.add_argument("--gene_list", type=str, nargs="+", default=None,
                        help='e.g. ["PTEN","TP53","RB1"]')
    parser.add_argument("--gene_mode", type=str, default="single",
                        choices=["single", "combination"],
                        help='"single": isp single gene; "combination": isp combination of multi-genes')
    parser.add_argument("--background", type=int, default=0,
                        help="0 表示不计算背景分布，>0 表示计算背景分布")

    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--endpoints", type=int, default=1)
    parser.add_argument("--ko_method", type=str, default=None)
    parser.add_argument("--ki_method", type=str, default=None)

    # Other params
    parser.add_argument("--cell_ratio", type=int, default=None)
    parser.add_argument("--pred_cells", type=str, default="internal",
                        help="internal prediction or external prediction for other proj")
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

    analyzer = ISP_GEP_Analyzer(args)
    analyzer.run()


# ============================================================================
# Test and Debug Entry
# ============================================================================


if __name__ == "__main__":
    # Check if running in interactive environment (e.g., Jupyter)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            # Interactive mode - for testing
            print("Running in interactive mode with test parameters...")
            args = argparse.Namespace(
                device="1",
                proj="xenium",
                model_ids=["GF_PR"],
                marker_set="SET1",
                gene_list=["PTEN", "TP53", "RB1"],
                gene_mode="combination",
                window=0,
                endpoints=1,
                ko_method="v1",
                ki_method=None,
                background=0,
                cell_ratio=5000,
                pred_cells="internal",
                do_force=False,
                work_dir="/dataSSD7T/liss/work/scPCa/model/"
            )
            analyzer = ISP_GEP_Analyzer(args)
            analyzer.run()
        else:
            # Script mode
            main()
    except NameError:
        # Script mode
        main()