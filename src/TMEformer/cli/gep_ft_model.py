#!/usr/bin/env python3
"""
TMEformer GEP Model Training Script

Fine-tune TMEformer model for multi-gene expression prediction.

freeze_layers parameter explanation:
    6   : Freeze the first 6 Transformer layers (32.77% parameters)
    -6  : Freeze the first 6 Transformer layers + input embedding layer (65.86% parameters) [default]
    99  : Freeze all BERT parameters (98.63% parameters)
"""

import argparse
import logging
import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from IPython import get_ipython

import TMEformer
from TMEformer import Regressor
from TMEformer.tme import TmeModeling_utils as tu
from TMEformer.tme.TmeModeling_utils import str2bool


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TrainingArgs:
    """Training arguments configuration."""
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 200
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    seed: int = 42
    logging_steps: int = 100
    disable_tqdm: bool = True
    optim: str = "adamw_torch"


@dataclass
class CellStateDict:
    """Cell state dictionary configuration."""
    state_key: str = "geps"
    states: str = "all"


@dataclass
class ProjectConfig:
    """Project configuration parameters."""
    proj: str
    gene_set: str
    train_mode: str
    model_id: str
    ft_model_id: str
    work_dir: str
    patch_size: int
    attr_col: str
    num_crossval_splits: int
    seed: int


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    freeze_layers: int
    use_quant: bool
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float


# ============================================================================
# GEP Trainer Class
# ============================================================================


class GEPTrainer:
    """Trainer for TMEformer GEP model fine-tuning."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.proj = args.proj
        self.gene_set = args.gene_set
        self.train_mode = args.train_mode
        self.model_id = args.model_id
        self.ft_model_id = args.ft_model_id
        self.freeze_layers = args.freeze_layers
        self.use_quant = args.use_quant
        self.work_dir = args.work_dir
        self.patch_size = args.patch_size
        self.attr_col = args.attr_col
        self.num_crossval_splits = args.num_crossval_splits
        self.seed = args.seed
        self.force_id = args.force_id

        self._setup_environment()
        self._setup_logger()
        self._load_model_genes()
        self._setup_paths()
        self._register_ft_id()

    def _setup_environment(self):
        """Setup CUDA environment."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    def _setup_logger(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger("GEPTrainer")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s [GEP] %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _load_model_genes(self):
        """Load model gene set."""
        with open(self.work_dir + "isp_gene_exp/MODEL_FT_GENE_SET.json", "r") as f:
            FT_Multi_SETs = json.load(f)
        self.model_genes = FT_Multi_SETs[self.gene_set]
        self.logger.info(f"Model gene set: {self.model_genes}")
        self.logger.info(f"Number of genes: {len(self.model_genes)}")


    def _register_ft_id(self):
        """Register fine-tuned model ID."""
        model_ft_id_map_path = Path(self.work_dir) / "isp_gene_exp/MODEL_FT_ID_MAP.json"
        with open(model_ft_id_map_path, "r") as f:
            model_ft_id_map = json.load(f)
        if self.ft_model_id in model_ft_id_map[self.proj][self.gene_set].keys():
            if not self.force_id:
                raise ValueError(f"FT Model ID {self.ft_model_id} already exists in MODEL_FT_ID_MAP.json")
        else:
            model_ft_id_map[self.proj][self.gene_set][self.ft_model_id] = self.model_id
            with open(model_ft_id_map_path, "w") as f:
                json.dump(model_ft_id_map, f, indent=4)

    def _setup_paths(self):
        """Setup paths and configurations."""
        if "xenium" in self.proj:
            self.models_dict = tu.generate_pr_models_dict("checkpoint_xe", self.work_dir)
        else:
            raise ValueError(f"Unsupported project: {self.proj}")

        self.model_directory = self.models_dict[self.model_id][0]
        self.data_version = self.models_dict[self.model_id][1]
        self.dataset_file = (
            Path(self.work_dir) /
            f"data/{self.proj}/datasets_ft/{self.data_version}/{self.proj}_epi_gep_{self.gene_set}.dataset"
        )

        self.p_output = Path(self.work_dir) / f"isp_gene_exp/{self.proj}/output_ft/{self.gene_set}/{self.ft_model_id}"
        self.p_output.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Model directory: {self.model_directory}")
        self.logger.info(f"Data version: {self.data_version}")
        self.logger.info(f"Dataset file: {self.dataset_file}")

    def _load_and_prepare_metadata(self) -> pd.DataFrame:
        """Load and prepare metadata."""
        obsmeta = pd.read_csv(Path(self.work_dir) / f"data/{self.proj}/processed/{self.proj}_obsmeta.csv")

        if f"patch_{self.patch_size}" not in obsmeta.columns:
            obsmeta = obsmeta.groupby("sample_id", group_keys=False).apply(
                tu.compute_patch_ids,
                patch_size=self.patch_size,
                include_groups=False
            )
            obsmeta = obsmeta.rename(columns={"patch_id": f"patch_{self.patch_size}"})

        return obsmeta

    def _create_regressor(self) -> Regressor:
        """Create Regressor object."""
        training_args = TrainingArgs(
            num_train_epochs=self.args.num_epochs,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            per_device_train_batch_size=self.args.train_batch_size,
            seed=self.seed,
            optim="adamw_torch" if not self.use_quant else "adamw_bnb_8bit"
        )

        cc = Regressor(
            regressor="cell",
            quantize=self.use_quant,
            cell_state_dict=CellStateDict().__dict__,
            training_args=training_args.__dict__,
            freeze_layers=self.freeze_layers,
            split_sizes={"train": 0.8, "valid": 0.2, "test": 0},
            num_crossval_splits=1 if "cv_attr" in self.train_mode else self.num_crossval_splits,
            forward_batch_size=self.args.eval_batch_size,
            nproc=16,
            token_dictionary_file=str(TMEformer.TOKEN_DICTIONARY_FILE),
            num_classes=len(self.model_genes)
        )
        cc.model_type = "CellClassifier_TME_MultiEXP"

        return cc

    def _prepare_dataset(self, cc: Regressor) -> Tuple[Any, str]:
        """Prepare dataset."""
        prepared_input_data_file = (
            Path(self.work_dir) /
            f"isp_gene_exp/{self.proj}/datasets/ft_{self.data_version}_{self.gene_set}_labeled.dataset"
        )

        if prepared_input_data_file.exists():
            self.logger.info("Skip prepare data - dataset already exists")
        else:
            self.logger.info("Preparing data...")
            cc.prepare_data(
                input_data_file=str(self.dataset_file),
                output_directory=str(prepared_input_data_file.parent),
                output_prefix=prepared_input_data_file.stem.replace("_labeled", "")
            )

        prepared_input_data = load_from_disk(str(prepared_input_data_file))

        return prepared_input_data, str(prepared_input_data_file)

    def _add_patch_ids(self, prepared_input_data: Any, obsmeta: pd.DataFrame) -> Tuple[Any, Optional[Dict]]:
        """Add patch IDs to dataset."""
        if "cv_attr" not in self.train_mode:
            return prepared_input_data, None

        if self.attr_col == "patch" and "patch_id" not in prepared_input_data.column_names:
            cell2patch_dict = dict(zip(obsmeta["cell_id"], obsmeta[f"patch_{self.patch_size}"]))

            def add_patch_id(example: Dict) -> Dict:
                return {"patch_id": cell2patch_dict[example["cell_id"]]}

            prepared_input_data = prepared_input_data.map(add_patch_id, num_proc=10)

        split_folds = tu.build_dataset_kfolds(
            prepared_input_data,
            attr_key=f"{self.attr_col}_id",
            n_splits=self.num_crossval_splits,
            seed=self.seed
        )

        return prepared_input_data, split_folds

    def _train_cross_validation(self, cc: Regressor, model_directory: str,
                                prepared_input_data: Any, prepared_input_data_file: str,
                                split_folds: Optional[Dict] = None) -> None:
        """Execute cross-validation training."""
        candi_name = "ray00"
        output_dir = self.p_output / candi_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if "cv_attr" not in self.train_mode:
            self.logger.info("Running random cross-validation...")
            cc.validate(
                model_directory=model_directory,
                prepared_input_data_file=prepared_input_data_file,
                output_directory=str(output_dir),
                output_prefix="cv"
            )
        else:
            self.logger.info("Running attr-independent cross-validation...")
            for i in split_folds.keys():
                self.logger.info(f"Split--{i}")
                model_path = output_dir / f"geneformer_cellRegressor_fold{i}" / "ksplit1" / "model.safetensors"

                if not model_path.exists():
                    cc.validate(
                        model_directory=model_directory,
                        prepared_input_data_file=prepared_input_data_file,
                        split_id_dict=split_folds[i],
                        output_directory=str(output_dir),
                        output_prefix=f"fold{i}"
                    )
                else:
                    self.logger.info(f"Skip fold{i} - model already exists")

            tu.transfer_ft_indep_dirs(str(output_dir))

    def _train_all_data(self, cc: Regressor, model_directory: str,
                        prepared_input_data: Any, prepared_input_data_file: str) -> None:
        """Train using all data."""
        train_all_dir = self.p_output / "train_all"
        train_all_dir.mkdir(parents=True, exist_ok=True)
        model_path = train_all_dir / "geneformer_cellRegressor_all" / "ksplit1" / "model.safetensors"

        if not model_path.exists():
            self.logger.info("Training on all data...")
            cc.train_all_data(
                model_directory=model_directory,
                prepared_input_data_file=prepared_input_data_file,
                output_directory=str(train_all_dir),
                output_prefix="all"
            )
        else:
            self.logger.info("Skip train_all - model already exists")

    def _save_gene_set(self) -> None:
        """Save gene set."""
        gene_set_path = self.p_output / "gene_set.pickle"
        with open(gene_set_path, "wb") as f:
            pickle.dump(self.model_genes, f)
        self.logger.info(f"Gene set saved to {gene_set_path}")

    def run(self) -> None:
        """Run the GEP training pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("TMEformer GEP Model Training")
        self.logger.info("=" * 80)

        self.logger.info(f"Project: {self.proj}")
        self.logger.info(f"Gene set: {self.gene_set}")
        self.logger.info(f"Train mode: {self.train_mode}")
        self.logger.info(f"Model ID: {self.model_id}")
        self.logger.info(f"Freeze layers: {self.freeze_layers}")

        obsmeta = self._load_and_prepare_metadata()

        self.logger.info("Creating regressor...")
        cc = self._create_regressor()

        self.logger.info("Loading/preparing dataset...")
        prepared_input_data, prepared_input_data_file = self._prepare_dataset(cc)

        if np.array(prepared_input_data["label"]).shape[1] != len(self.model_genes):
            raise ValueError(
                f"Number of genes mismatch: "
                f"MODEL_GENEs={len(self.model_genes)}, "
                f"prepared_input_data={np.array(prepared_input_data['label']).shape[1]}"
            )

        prepared_input_data, split_folds = self._add_patch_ids(prepared_input_data, obsmeta)

        self.logger.info(f"Output directory: {self.p_output}")

        if "cv" in self.train_mode:
            self._train_cross_validation(
                cc, self.model_directory, prepared_input_data, prepared_input_data_file, split_folds
            )

        if "all" in self.train_mode:
            self._train_all_data(cc, self.model_directory, prepared_input_data, prepared_input_data_file)

        self._save_gene_set()

        self.logger.info("=" * 80)
        self.logger.info("Training completed successfully!")
        self.logger.info("=" * 80)


# ============================================================================
# Argument Parser
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Train TMEformer model for multi-gene expression prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training modes:
  cv           : Random cross-validation
  cv_all       : Random cross-validation + all data modeling
  cv_attr      : Attr-independent cross-validation
  cv_attr_all  : Attr-independent cross-validation + all data modeling
  all          : All data modeling

freeze_layers parameter:
  6   : Freeze first 6 Transformer layers (32.77% parameters)
  -6  : Freeze first 6 Transformer layers + input embedding (65.86% parameters) [default]
  99  : Freeze all BERT parameters (98.63% parameters)

Examples:
  # Basic usage
  %(prog)s --proj xenium --gene_set SET0 --train_mode cv

  # Custom hyperparameters
  %(prog)s --proj xenium_10X_bca --gene_set SET3 \\
      --learning_rate 5e-5 --weight_decay 0.005 --num_epochs 3

  # Attr-independent cross-validation
  %(prog)s --proj xenium --train_mode cv_attr_all --attr_col patch
        """
    )

    # Basic parameters
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--proj", type=str, default="xenium",
        help="Project name (default: xenium)"
    )

    # Model parameters
    parser.add_argument(
        "--model_id", type=str, default="GF_PR",
        help="Base model ID (default: GF_PR)"
    )
    parser.add_argument(
        "--ft_model_id", type=str, default="FT9999_99",
        help="Fine-tuned model ID (default: FT9999_99)"
    )
    parser.add_argument(
        "--freeze_layers", type=int, default=-6,
        help="Number of layers to freeze (default: -6)"
    )
    parser.add_argument(
        "--gene_set", type=str, default="SET1",
        help="Gene set identifier (default: SET1)"
    )

    # Training parameters
    parser.add_argument(
        "--train_mode", type=str, default="cv",
        choices=["cv", "cv_all", "cv_attr", "cv_attr_all", "all"],
        help="Training mode (default: cv)"
    )
    parser.add_argument(
        "--attr_col", type=str, default="patch", choices=["sample", "patch"],
        help="Attribute column for attr-independent CV (default: patch)"
    )
    parser.add_argument(
        "--use_quant", type=str2bool, default=False,
        help="Use quantization (8bit) for training (default: False)"
    )

    # Hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16,
        help="Training batch size (default: 16)"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=32,
        help="Evaluation batch size (default: 32)"
    )

    # Data parameters
    parser.add_argument(
        "--work_dir", type=str, default="/dataSSD7T/liss/work/scPCa/model/",
        help="Working directory path"
    )
    parser.add_argument(
        "--patch_size", type=int, default=2000,
        help="Patch size for spatial partitioning (default: 2000)"
    )
    parser.add_argument(
        "--num_crossval_splits", type=int, default=5,
        help="Number of cross-validation splits (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--force_id", type=str2bool, default=False, 
        help="Ignore existing ft-model ID"
    )

    return parser


# ============================================================================
# CLI Main Entry
# ============================================================================


def main():
    """CLI main entry function."""
    parser = create_parser()
    args = parser.parse_args()

    trainer = GEPTrainer(args)
    trainer.run()


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
                model_id="GF_PR",
                ft_model_id="FT1010_01",
                freeze_layers=-6,
                gene_set="SET3",
                train_mode="cv_attr_all",
                attr_col="patch",
                use_quant=False,
                learning_rate=1e-4,
                weight_decay=0.01,
                num_epochs=1,
                train_batch_size=16,
                eval_batch_size=32,
                work_dir="/dataSSD7T/liss/work/scPCa/model/",
                patch_size=2000,
                num_crossval_splits=5,
                seed=42,
                local_rank=-1
            )
            trainer = GEPTrainer(args)
            trainer.run()
        else:
            # Script mode
            main()
    except NameError:
        # Script mode
        main()