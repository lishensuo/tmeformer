#!/usr/bin/env python3
"""
Gene Expression Perturbation Analysis Script

Predict the change of marker gene expression for TME-Rank/Composition ISP.
"""

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_from_disk
from IPython import get_ipython

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
    window: float = 0
    endpoints: int = 1
    ko_method: Optional[str] = None
    ki_method: Optional[str] = None


@dataclass
class ProjectConfig:
    """Project configuration parameters."""
    name: str
    max_isp_cells: int


PROJECT_CONFIGS: Dict[str, ProjectConfig] = {
    "xenium": ProjectConfig(name="xenium", max_isp_cells=10000),
    "pca165": ProjectConfig(name="pca165", max_isp_cells=2000),
    "xenium_10X_bca": ProjectConfig(name="xenium_10X_bca", max_isp_cells=10000),
    "xenium_10X_bca2": ProjectConfig(name="xenium_10X_bca2", max_isp_cells=10000),
    "xenium_10X_cc": ProjectConfig(name="xenium_10X_cc", max_isp_cells=10000),
    "xenium_10X_ov": ProjectConfig(name="xenium_10X_ov", max_isp_cells=10000),
}


# ============================================================================
# ISP TME GEP Analyzer Class
# ============================================================================


class ISP_TME_GEP_Analyzer:
    """Analyzer for marker gene expression perturbation analysis."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.proj = args.proj
        self.proj_external = args.proj_external
        self.model_ids = args.model_id
        self.marker_set = args.marker_set
        self.marker_id = args.marker_id

        self.tme_method = args.tme_method
        self.cell_cluster_file = args.cell_cluster_file
        self.isp_cluster = args.isp_cluster
        self.endpoints = args.endpoints
        self.window = args.window
        self.fixed_clusters = args.fixed_clusters
        self.isp_genes = args.gene_list
        self.gene_mode = args.gene_mode

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

        if self.tme_method == "rank":
            if self.gene_mode == "combination":
                self.isp_gene_list = [sorted(self.isp_genes)]
            elif self.gene_mode == "single":
                self.isp_gene_list = [[gene] for gene in self.isp_genes]
        elif self.tme_method == "composition":
            self.isp_gene_list = [[None]]

        if self.tme_method == "composition":
            self.isp_config = ISPConfig(window=self.window, endpoints=self.endpoints)
        elif self.tme_method == "rank":
            self.isp_config = ISPConfig(window=0, endpoints=1)

        if self.cell_ratio is not None:
            self.proj_config.max_isp_cells = self.cell_ratio

    def _load_cell_cluster_dict(self) -> None:
        """Load cell cluster dictionary."""
        with open(Path(self.work_dir) / f"data/{self.proj}/processed/{self.proj}_tme_id_dict.pkl", "rb") as f:
            main_tme_id_dict = pickle.load(f)
        main_tme_ids = set(main_tme_id_dict.values())

        isp_proj = self.proj if self.pred_cells == "internal" else self.pred_cells

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

    def _load_model_config(self) -> None:
        """Load model configuration."""
        config_path = Path(self.work_dir) / "isp_gene_exp/MODEL_FT_ID_MAP.json"
        with open(config_path, "r") as f:
            model_id_json = json.load(f)

        try:
            model_id_dict = model_id_json[self.proj][self.marker_set]
        except KeyError:
            available_keys = list(model_id_json[self.proj].keys())
            raise KeyError(f"Valid marker set: {available_keys}")

        self.model_id_dict = {v: k for k, v in model_id_dict.items()}

        if self.model_ids is not None:
            for model_id in self.model_ids:
                if model_id not in self.model_id_dict:
                    raise ValueError(f"Invalid model id: {model_id}")
            self.model_id_dict = {k: v for k, v in self.model_id_dict.items() if k in self.model_ids}

    def _setup_logger(self) -> None:
        """Setup logging configuration."""
        config_suffix = tu_isp_ds.isp_run_suffix(self.isp_config.__dict__, isp="tme")

        self.tme_method_label = {
            "composition": "TME_Composition",
            "rank": "TME_Rank",
        }[self.tme_method]

        if self.marker_id is None:
            self.pre_save_name = (
                f"{self.marker_set}{self.file_suffix}-{self.tme_method_label}-"
                f"S{self.proj_config.max_isp_cells}-{config_suffix}"
            )
        else:
            self.pre_save_name = (
                f"{self.marker_set}_{self.marker_id}{self.file_suffix}-{self.tme_method_label}-"
                f"S{self.proj_config.max_isp_cells}-{config_suffix}"
            )

        log_name = self._make_full_save_name(
            self.pre_save_name, self.tme_method, self.isp_cluster,
            self.fixed_clusters, self.isp_gene_list
        )
        log_dir = Path(self.work_dir) / "isp_gene_exp" / self.proj / "log_isp" / self.marker_set / f"tme_{self.tme_method}"
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.custom_log_suffix:
            log_name += "_" + self.custom_log_suffix

        logfile = log_dir / f"{log_name}.log"

        self.logger = logging.getLogger(f"{log_name}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(logfile, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP TME GEP] %(message)s")
            )
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [ISP TME GEP] %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _make_full_save_name(self, pre_save_name: str,
                             tme_method: str,
                             isp_cluster: int,
                             fixed_clusters: Optional[List[int]],
                             isp_gene: Optional[List] = None) -> str:
        """Make full save name for output files."""
        if tme_method == "rank":
            if isinstance(isp_gene[0], list):
                if len(isp_gene) > 5:
                    isp_gene_text = '_'.join(['+'.join(gene) for gene in isp_gene[:5]])
                    isp_gene_text += f"_and_{len(isp_gene) - 5}more"
                else:
                    isp_gene_text = '_'.join(['+'.join(gene) for gene in isp_gene])
            elif isinstance(isp_gene[0], str):
                isp_gene_text = '+'.join(isp_gene)
            return f"{pre_save_name}-TME{isp_cluster}_{isp_gene_text}"
        elif tme_method == "composition":
            fixed_label = "all" if fixed_clusters is None else '+'.join(str(i) for i in fixed_clusters)
            return f"{pre_save_name}-TME{isp_cluster}_FIX{fixed_label}"
        return pre_save_name

    def _load_model_and_dataset(self, model_id: str, ft_model_id: str) -> Tuple[Any, Any]:
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

        if hasattr(model.config, "tme_config") and self.pred_cells != "internal":
            model.bert.cell_bert.tme_config.gf_tme_emb_path = f"data/{self.pred_cells}/gf_emb/GF_CL_L0/"

        PR_XE_MODELS_DICT = tu.generate_pr_models_dict("checkpoint_xe", self.work_dir)
        version = PR_XE_MODELS_DICT[model_id][1]
        dataset_path = (
            Path(self.work_dir) / "isp_gene_exp" / self.proj / "datasets" /
            f"isp_{version}_{self.marker_set}{self.file_suffix}.dataset"
        )

        dataset = load_from_disk(str(dataset_path))
        dataset = tu.modify_tme_dataset(model, dataset)
        dataset = tu_isp_ds.add_cluster_types(dataset, self.isp_cluster, self.cell_cluster_dict)

        return model, dataset

    def _process_tme_isp(self, model: Any, dataset: Any, model_id: str, save_dir: Path) -> None:
        """Process TME ISP analysis."""
        for j, isp_gene in enumerate(self.isp_gene_list):
            save_path = save_dir / (
                self._make_full_save_name(self.pre_save_name, self.tme_method, self.isp_cluster,
                                          self.fixed_clusters, isp_gene) + ".csv"
            )

            self.logger.info(
                f"## Start model {model_id} | ISP {isp_gene} ({j + 1}/{len(self.isp_gene_list)})"
            )
            self.logger.info(f"==> Save PATH: {save_path}")

            if not self.do_force and save_path.exists():
                self.logger.info("==> Skip: Found existing result file")
                continue

            score_df_list = []
            if isp_gene[0] is None:
                isp_gene_token = isp_gene
            else:
                isp_gene_token = [tu.symbol2token(gene) for gene in isp_gene]

            for idx, model_gene in enumerate(self.model_genes):
                # 当指定预测的model gene时, 不处理set里的其它model gene
                if self.marker_id is not None and idx != self.marker_id:
                    continue
                dataset_filtered = self._filter_dataset_for_tme(
                    dataset, self.tme_method, self.isp_cluster, self.fixed_clusters,
                    isp_gene_token, self.proj_config.max_isp_cells
                )

                self.logger.info(
                    f"==> ISP cells: {len(dataset_filtered)} / {len(dataset)} "
                    f"for {model_gene}"
                )

                score_df = tu_isp_gep.calc_gep_tme_isp_score_from_cells(
                    dataset_filtered, self.tme_method, self.isp_cluster, model,
                    self.cell_tme_dict, self.cell_cluster_dict, self.isp_config.__dict__,
                    batch=8, pred_idx=idx, fixed_cluster_ids=self.fixed_clusters,
                    gene_symbol=isp_gene, resample=True, logger=self.logger,
                    work_dir=self.work_dir, proj=self.proj
                )
                score_df["model_gene"] = model_gene
                score_df_list.append(score_df)

            cells_isp_score_df = pd.concat(score_df_list)
            cells_isp_score_df.to_csv(save_path)

    def _filter_dataset_for_tme(self, dataset: Any,
                                tme_method: str,
                                isp_cluster: int,
                                fixed_clusters: Optional[List[int]],
                                isp_gene_token: List[str],
                                max_cells: int) -> Any:
        """Filter dataset for TME ISP analysis."""
        dataset_filtered = dataset

        if tme_method == "composition":
            dataset_filtered = tu_isp_ds.filter_tme_composition_dataset(
                dataset_filtered, isp_cluster_id=isp_cluster, fixed_cluster_ids=fixed_clusters,
                existed=True, ratio=max_cells, cluster_endpoint=self.isp_config.endpoints
            )

        elif tme_method == "rank":
            dataset_filtered = tu_isp_ds.filter_tme_composition_dataset(
                dataset_filtered, isp_cluster_id=isp_cluster, existed=True
            )
            dataset_filtered = tu_isp_ds.filter_tme_rank_dataset(
                dataset_filtered, isp_cluster_id=isp_cluster,
                cell_cluster_dict=self.cell_cluster_dict,
                proj=self.proj if self.pred_cells == "internal" else self.pred_cells,
                tokens=isp_gene_token, existed=True, ratio=max_cells, work_dir=self.work_dir
            )

        return dataset_filtered

    def run(self) -> None:
        """Run the ISP analysis pipeline."""
        self._load_cell_cluster_dict()
        self._load_model_config()
        self._setup_logger()

        self.logger.info(f"# TASK   : {self.logger.name}")
        self.logger.info(f"# Params : {self.args.__dict__}")
        self.logger.info(f"# Models : {self.model_id_dict}")

        for i, (model_id, ft_model_id) in enumerate(self.model_id_dict.items()):
            self.logger.info(f"## Start model {model_id} ({i + 1}/{len(self.model_id_dict)})")

            model, dataset = self._load_model_and_dataset(model_id, ft_model_id)

            save_dir = (
                Path(self.work_dir) / "isp_gene_exp" / self.proj / "output_isp" /
                model_id / self.marker_set / f"tme_{self.tme_method}"
            )
            save_dir.mkdir(parents=True, exist_ok=True)

            self._process_tme_isp(model, dataset, model_id, save_dir)


# ============================================================================
# Argument Parser
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="ISP_TME_GEP_Analyzer")

    # Basic parameters
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--proj", type=str, default="xenium",
                        choices=["xenium", "xenium_10X_bca", "xenium_10X_bca2",  "xenium_10X_cc", "xenium_10X_ov"])
    parser.add_argument("--proj_external", type=str, default=None)
    parser.add_argument("--model_id", type=str, nargs="+", default=None,
                        help='e.g. ["GF_PR", "GF_CL"]')
    parser.add_argument("--marker_set", type=str, default=None,
                        help='e.g. "SET0"')
    parser.add_argument("--marker_id", type=int, default=None,
                        help='None for all markers or index id for one marker in the set'), 

    # ISP parameters
    parser.add_argument("--tme_method", type=str, default=None,
                        choices=["composition", "rank"])
    parser.add_argument("--cell_cluster_file", type=str, default="main")
    parser.add_argument("--isp_cluster", type=int, default=None)
    parser.add_argument("--fixed_clusters", type=int, nargs="+", default=None)
    parser.add_argument("--endpoints", type=float, default=1)
    parser.add_argument("--window", type=float, default=0)
    parser.add_argument("--gene_list", type=str, nargs="+", default=None,
                        help='e.g. ["PTEN","TP53","RB1"]')
    parser.add_argument("--gene_mode", type=str, default="single",
                        choices=["single", "combination"],
                        help='"single": isp single gene; "combination": isp combination of multi-genes')

    # Other parameters
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

    analyzer = ISP_TME_GEP_Analyzer(args)
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
                proj="xenium_10X_bca",
                proj_external=None,
                model_id=["GF_D1021_05"],
                marker_set="SET3",
                tme_method="rank",
                cell_cluster_file="main",
                isp_cluster=2,
                fixed_clusters=None,
                endpoints=1,
                window=0,
                gene_list=["IL6"],
                gene_mode="single",
                cell_ratio=5000,
                pred_cells="internal",
                do_force=False,
                custom_log_suffix=None,
                work_dir="/dataSSD7T/liss/work/scPCa/model/"
            )
            analyzer = ISP_TME_GEP_Analyzer(args)
            analyzer.run()
        else:
            # Script mode
            main()
    except NameError:
        # Script mode
        main()