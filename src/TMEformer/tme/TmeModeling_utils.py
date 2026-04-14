"""
TMEformer TME Modeling Utilities
================================

This module provides utility functions for TME modeling tasks, including:
- Data preprocessing and manipulation
- Model prediction and evaluation
- Dataset splitting and cross-validation
- Gene symbol to token conversion
- TensorBoard log parsing
- Model checkpoint management

Key Functions:
    - get_all_valid_genes: Get valid genes for a project
    - symbol2token: Convert gene symbols to token IDs
    - model_predict: Make predictions with trained models
    - build_dataset_kfolds: Split dataset into k-folds
    - modify_tme_dataset: Rename TME columns based on model config
    - read_tensorboard_log: Parse TensorBoard event files
    - get_pr_model_log: Get pretraining model logs
    - get_ft_model_log: Get fine-tuning model logs
"""

import argparse
import json
import os
import pickle
import re
import random
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar
from sklearn.model_selection import KFold
from tensorboard.backend.event_processing import event_accumulator
from tqdm.auto import trange

from .. import ENSEMBL_MAPPING_FILE, TOKEN_DICTIONARY_FILE
from .TmeModeling_bert import (
    TmeBertForCellClassification,
    TmeBertForMultiGeneExpressionPrediction,
    TmeBertForSequenceClassification,
)


# =============================================================================
# Gene and Token Utilities
# =============================================================================


def str2bool(value):
    """
    Transform string to boolean for script parameters.
    
    Args:
        value: String or boolean value
    
    Returns:
        Boolean value
    
    Raises:
        argparse.ArgumentTypeError: If value cannot be converted to boolean
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "yes", "y", "1"}:
        return True
    elif value.lower() in {"false", "f", "no", "n", "0"}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_all_valid_genes(proj: str, work_dir: Optional[str] = None) -> np.ndarray:
    """
    Get all valid genes for a project.

    Args:
        proj: Project identifier (e.g., "xenium", "pca165").
        work_dir: Working directory path. If None, uses current directory.

    Returns:
        Array of valid gene names (those with valid TOKEN IDs).

    Raises:
        ValueError: If gene ID file not found.

    Example:
        >>> genes = get_all_valid_genes("xenium", work_dir="/path/to/work/")
    """
    if work_dir is None:
        work_dir = "./"

    gene_ids_file = os.path.join(work_dir, f"data/{proj}/processed/{proj}_gene_ids.csv")
    if not os.path.exists(gene_ids_file):
        raise ValueError(f"Gene ID file not found: {gene_ids_file}")

    genes = pd.read_csv(gene_ids_file)
    genes_valid = genes[genes["TOKEN"] != -999]["Gene"].values
    return genes_valid


def symbol2token(gene_symbol: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Transform gene symbol to token ID.

    Args:
        gene_symbol: Single gene symbol string or list of gene symbols.

    Returns:
        Token ID (int) or list of token IDs.

    Raises:
        ValueError: If gene symbol not found in token dictionary.
        FileNotFoundError: If token dictionary or mapping file not found.

    Example:
        >>> token = symbol2token("TP53")
        >>> tokens = symbol2token(["TP53", "BRCA1", "EGFR"])
    """
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    with open(ENSEMBL_MAPPING_FILE, "rb") as f:
        ensembl_mapping_dict = pickle.load(f)

    if isinstance(gene_symbol, str):
        gene_token = gene_token_dict.get(ensembl_mapping_dict.get(gene_symbol))
        if gene_token is None:
            raise ValueError(
                f"Gene symbol ({gene_symbol}) not found in the token dictionary."
            )
        return gene_token

    if isinstance(gene_symbol, list):
        gene_tokens = [
            gene_token_dict.get(ensembl_mapping_dict.get(symbol))
            for symbol in gene_symbol
        ]
        if None in gene_tokens:
            gene_to_check = gene_symbol[gene_tokens.index(None)]
            raise ValueError(
                f"Some gene symbol ({gene_to_check}) not found in the token dictionary."
            )
        return gene_tokens

    raise TypeError("gene_symbol must be a string or list of strings.")


def delete_multi_gene_token_and_add_gep(
    example: Dict[str, Any],
    gene_token_ids: List[int],
    gep_dict: Dict[str, List[float]],
    mask_ratio: float = 1
) -> Dict[str, Any]:
    """
    Add the GEP column and retokenize the input_ids by removing specified gene tokens.

    Args:
        example: Dataset example dictionary containing input_ids, length, cell_id.
        gene_token_ids: List of gene token IDs to remove from input_ids.
        gep_dict: Dictionary mapping cell_id to gene expressions.
        mask_ratio: Ratio of marker tokens to mask.

    Returns:
        Updated example dictionary with modified input_ids and added geps column.


    """
    input_ids = example["input_ids"]
    cell_id = example["cell_id"]

    gene_exps = gep_dict[cell_id]
    example["geps"] = [-1 if gene_exp == 0 else gene_exp for gene_exp in gene_exps]

    present_markers = list(set(input_ids) & set(gene_token_ids))
    if mask_ratio == 1:
        masked_tokens = present_markers
    else:
        masked_tokens = set(random.sample(present_markers, int(len(present_markers) * mask_ratio)))

    updated_input_ids = [
        input_id for input_id in input_ids if input_id not in masked_tokens
    ]
    updated_length = len(updated_input_ids)

    example["input_ids"] = updated_input_ids
    example["length"] = updated_length

    return example


# =============================================================================
# Model Prediction Utilities
# =============================================================================


def model_predict(
    model: torch.nn.Module,
    evalset: Union[Dataset, str],
    forward_batch_size: int = 32,
    verbose: bool = False,
    **kwargs: Any,
) -> tuple:
    """
    Make predictions with a trained model.

    Args:
        model: Trained model (TmeBertFor* class).
        evalset: Dataset object or path to dataset directory.
        forward_batch_size: Batch size for forward pass.
        verbose: Whether to show progress bar.
        **kwargs: Additional arguments (e.g., tme_cell_embs).

    Returns:
        Tuple containing model predictions (logits).

    Raises:
        ValueError: If evalset is not valid or dimensions don't match.

    Note:
        Supports models with TME integration (use_tme=True) by automatically
        handling TME cell embeddings and types.
    """
    from .. import evaluation_utils as eu

    model.eval()

    if isinstance(evalset, str):
        evalset = load_from_disk(evalset)
    if not isinstance(evalset, Dataset):
        raise ValueError(
            "evalset must be a Dataset object or a path to a dataset directory."
        )

    evalset_len = len(evalset)
    max_evalset_len = max(
        evalset.select([i for i in range(evalset_len)])["length"]
    )

    disable_progress_bar()

    tme_cell_embs = kwargs.get("tme_cell_embs")
    if tme_cell_embs is not None:
        if not isinstance(tme_cell_embs, torch.Tensor):
            tme_cell_embs = torch.tensor(tme_cell_embs)
        if tme_cell_embs.shape[0] != len(evalset):
            raise ValueError("tme_cell_embs.shape[0] != len(input_dataset)")

    logits_batches = []
    reg_batches = []

    for i in trange(0, evalset_len, forward_batch_size, disable=not verbose):
        max_range = min(i + forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])

        padded_batch = eu.preprocess_classifier_batch(batch_evalset, max_evalset_len)
        padded_batch.set_format(type="torch")
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]

        if hasattr(model.config, "use_tme") and model.config.use_tme:
            padded_batch = modify_tme_dataset(model, padded_batch)

            tme_cells_minibatch = padded_batch["tme_cells"]
            tme_types_minibatch = padded_batch["tme_types"]
            cell_id_minibatch = padded_batch["cell_id"]
            sample_id_minibatch = padded_batch["sample_id"]

            if tme_cell_embs is not None:
                tme_cell_embs_minibatch = tme_cell_embs[range(i, max_range)]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_data_batch.to(model.device),
                    attention_mask=attn_msk_batch.to(model.device),
                    cell_id=cell_id_minibatch.to(model.device),
                    sample_id=sample_id_minibatch.to(model.device),
                    tme_cells=tme_cells_minibatch.to(model.device),
                    tme_cell_embs=(
                        tme_cell_embs_minibatch.to(model.device)
                        if tme_cell_embs is not None
                        else None
                    ),
                    tme_types=tme_types_minibatch.to(model.device),
                )
        else:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_data_batch.to(model.device),
                    attention_mask=attn_msk_batch.to(model.device),
                )

        if "logits" in outputs:
            logits = outputs["logits"].detach().cpu()
            if logits.dim() == 1 or logits.shape[1] == 1:
                logits = logits.reshape(-1, 1)
            logits_batches.append(logits)
        if "reg_output" in outputs:
            reg_batches += [outputs["reg_output"].detach().cpu().reshape(-1, 1)]

    if isinstance(
        model,
        (
            TmeBertForSequenceClassification,
            TmeBertForCellClassification,
            TmeBertForMultiGeneExpressionPrediction,
        ),
    ):
        predict_logits = torch.cat(logits_batches)
        return (predict_logits,)

    return outputs


def modify_tme_dataset(
    model: torch.nn.Module, dataset: Dataset
) -> Dataset:
    """
    Rename TME columns in dataset based on model configuration.

    Args:
        model: Model with TME configuration.
        dataset: Dataset to modify.

    Returns:
        Modified dataset with renamed columns (tme_cells, tme_types).

    Raises:
        ValueError: If unsupported tme_level is provided.

    Note:
        The function renames columns like 'tme_cells256' to 'tme_cells'
        based on the model's tme_config.
    """
    if hasattr(model.config, "use_tme") and model.config.use_tme:
        if "tme_cells" in dataset.column_names and "tme_types" in dataset.column_names:
            return dataset

        tme_level = model.config.tme_config.get("tme_level", "cell")

        if tme_level == "cell":
            tme_size = model.config.tme_config["max_position_embeddings"]
        elif tme_level == "cluster":
            tme_size = 256
        else:
            raise ValueError(f"Unsupported tme_level: {tme_level}")

        column_mapping = {
            f"tme_cells{tme_size}": "tme_cells",
            f"tme_types{tme_size}": "tme_types",
        }

        dataset = dataset.rename_columns(column_mapping)

    return dataset


# =============================================================================
# Dataset Utilities
# =============================================================================


def build_dataset_kfolds(
    dataset: Dataset,
    attr_key: str = "sample_id",
    n_splits: int = 4,
    seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """
    Split dataset into k-folds with attribute independence.

    Args:
        dataset: Dataset to split.
        attr_key: Key for grouping attribute (e.g., "sample_id").
        n_splits: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping fold index to train/eval split information.

    Example:
        >>> folds = build_dataset_kfolds(dataset, attr_key="sample_id", n_splits=5)
        >>> fold_0 = folds[0]
        >>> train_samples = fold_0["train"]
        >>> eval_samples = fold_0["eval"]
    """
    unique_items = np.unique(dataset[attr_key])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sample_folds = {}

    for fold_idx, (train_idx, eval_idx) in enumerate(kf.split(unique_items)):
        train_items = unique_items[train_idx].tolist()
        eval_items = unique_items[eval_idx].tolist()

        attr_values = np.array(dataset[attr_key])
        train_cell_count = np.isin(attr_values, train_items).sum()
        eval_cell_count = np.isin(attr_values, eval_items).sum()

        print(
            f"[Fold {fold_idx}] train items: {len(train_items)}, cells: {train_cell_count}; "
            f"eval samples: {len(eval_items)}, cells: {eval_cell_count}"
        )

        sample_folds[fold_idx] = {
            "attr_key": attr_key,
            "train": train_items,
            "eval": eval_items,
        }

    return sample_folds


def compute_patch_ids(group: pd.DataFrame, patch_size: int = 500) -> pd.DataFrame:
    """
    Split spatial coordinates into patches.

    Args:
        group: DataFrame with spatial coordinates.
            Required columns: ["sample_id", "spatial_1", "spatial_2"]
            or ["sample_id", "x_centroid", "y_centroid"]
        patch_size: Size of each patch in coordinate units.

    Returns:
        DataFrame with added patch_id column.

    Note:
        patch_id format: "{sample_id}_{patch_x}_{patch_y}"
    """
    sample = group.name
    group = group.copy()

    if "x_centroid" in group.columns:
        group["spatial_1"] = group["x_centroid"]
    if "y_centroid" in group.columns:
        group["spatial_2"] = group["y_centroid"]

    group["patch_x"] = (group["spatial_1"] // patch_size).astype(int)
    group["patch_y"] = (group["spatial_2"] // patch_size).astype(int)

    group["sample_id"] = sample
    group["patch_id"] = (
        str(sample) + "_" + group["patch_x"].astype(str) + "_" + group["patch_y"].astype(str)
    )
    group = group.drop(columns=["patch_x", "patch_y"])
    return group


def get_tme_type_freq(
    proj: str = "xenium",
    set_id: str = "SET1",
    ds_version: str = "v3",
    ds_suffix: str = "_random",
    tme_size: int = 256,
    work_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate TME compositions for each target cell.

    Args:
        proj: Project name.
        set_id: Set identifier.
        ds_version: Dataset version.
        ds_suffix: Dataset suffix.
        tme_size: Size of TME neighborhood.
        work_dir: Working directory path.

    Returns:
        DataFrame with TME type frequencies (melted format).
            Columns: cell_id, tme_type, ratio
    """
    if work_dir is None:
        work_dir = "./"

    obsmeta = pd.read_csv(
        os.path.join(work_dir, f"data/{proj}/processed/{proj}_obsmeta.csv")
    )

    celltype_dict = (
        obsmeta[["cell_type", "tme_id"]]
        .drop_duplicates()
        .set_index("tme_id")
        .sort_index()["cell_type"]
        .to_dict()
    )

    tme_ids = list(celltype_dict.keys())

    ds_path = os.path.join(
        work_dir,
        f"isp_gene_exp/{proj}/datasets/ft_{ds_version}_{set_id}_labeled{ds_suffix}.dataset",
    )
    dataset = load_from_disk(ds_path)

    dataset.set_format(type="numpy", columns=[f"tme_types{tme_size}"])
    tme_cells_array = dataset[f"tme_types{tme_size}"]
    n_cells = tme_cells_array.shape[0]
    freq_result = np.zeros((n_cells, max(tme_ids) + 1))

    for i in range(n_cells):
        counts = np.bincount(tme_cells_array[i], minlength=freq_result.shape[1])
        freq_result[i] = counts / tme_size

    freq_result_df = pd.DataFrame(freq_result)
    freq_result_df = freq_result_df.loc[:, (freq_result_df != 0).any(axis=0)]
    freq_result_df = freq_result_df.rename(columns=celltype_dict)
    freq_result_df["cell_id"] = dataset["cell_id"]
    freq_result_df = freq_result_df.melt(id_vars=["cell_id"], var_name="tme_type", value_name="ratio")

    return freq_result_df


# =============================================================================
# ISP/TME Background Analysis Utilities
# =============================================================================


def merge_isp_tme_rank_bg(
    task_type: str = "gene_exp",
    parent_dir: str = "isp_gene_exp/xenium/output_isp/GF_D0528_06/SET1/tme_rank/",
    output_prefix: str = "SET1_random",
    isp_config: str = "W0_EP1_KO0_KI0",
    bg_size_each: int = 100,
    tme_id: int = 2,
    bg_genes: Optional[List[str]] = None,
    do_force: bool = False,
    remove_raw: bool = False,
    work_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Merge background ISP files from TME rank ISP on random genes.

    Args:
        task_type: Type of task ("gene_exp" or "emb_sim").
        parent_dir: Parent directory path (relative to work_dir).
        output_prefix: Prefix for output files.
        isp_config: ISP configuration string.
        bg_size_each: Number of samples per background gene.
        tme_id: TME identifier.
        bg_genes: List of background genes.
        do_force: Force overwrite existing files.
        remove_raw: Remove raw files after merging.
        work_dir: Working directory path.

    Returns:
        Merged DataFrame, or None if output file already exists.

    Raises:
        Exception: If all background files are missing.
    """
    if work_dir is None:
        work_dir = "./"

    parent_dir = os.path.join(work_dir, parent_dir)

    if bg_genes is None:
        bg_genes = []

    if task_type == "gene_exp":
        bg_total_cells = len(bg_genes) * bg_size_each
        bg_output_filename = (
            f"{output_prefix}-TME_Rank-S{bg_total_cells}-{isp_config}-"
            f"TME{tme_id}_Background_C{len(bg_genes[0].split('+'))}.csv"
        )
        bg_output_path = os.path.join(parent_dir, bg_output_filename)

        if os.path.exists(bg_output_path) and not do_force:
            print(f"{bg_output_path} exists")
            return None

        bg_fl_prefix = f"{output_prefix}-TME_Rank-S{bg_size_each}-{isp_config}-TME{tme_id}"
        bg_fls = [
            os.path.join(parent_dir, f"{bg_fl_prefix}_{bg_gene}.csv")
            for bg_gene in bg_genes
        ]

        failed = [bg_fl for bg_fl in bg_fls if not os.path.exists(bg_fl)]
        if len(failed) == len(bg_fls):
            raise Exception("All bg files not exist")
        if len(failed) > 0:
            print(f"{len(failed)}/{len(bg_fls)} bg results not exist")

        bg_gene_df_list = []
        for bg_fl, bg_gene in zip(bg_fls, bg_genes):
            if os.path.exists(bg_fl):
                bg_gene_df = pd.read_csv(bg_fl).drop(columns=["Unnamed: 0"])
                bg_gene_df["isp_gene"] = bg_gene
                bg_gene_df_list.append(bg_gene_df)

        bg_gene_df_merge = pd.concat(bg_gene_df_list)
        bg_gene_df_merge.to_csv(bg_output_path)

    elif task_type == "emb_sim":
        bg_total_cells = len(bg_genes) * bg_size_each
        bg_output_filename = (
            f"{output_prefix}-TME_Rank-L1-S{bg_total_cells}-{isp_config}-"
            f"TME{tme_id}_Background_C{len(bg_genes[0].split('+'))}.csv"
        )
        bg_output_path = os.path.join(parent_dir, bg_output_filename)

        if os.path.exists(bg_output_path) and not do_force:
            print(f"{bg_output_path} exists")
            return None

        bg_fl_prefix = f"{output_prefix}-TME_Rank-L1-S{bg_size_each}-{isp_config}-TME{tme_id}"
        bg_fls = [
            os.path.join(parent_dir, f"{bg_fl_prefix}_{bg_gene}.csv")
            for bg_gene in bg_genes
        ]

        failed = [bg_fl for bg_fl in bg_fls if not os.path.exists(bg_fl)]
        if len(failed) == len(bg_fls):
            raise Exception("All bg files not exist")
        if len(failed) > 0:
            print(f"{len(failed)}/{len(bg_fls)} bg results not exist")

        bg_gene_df_list = []
        for bg_fl, bg_gene in zip(bg_fls, bg_genes):
            if os.path.exists(bg_fl):
                bg_gene_df = pd.read_csv(bg_fl)
                bg_gene_df["isp_gene"] = bg_gene
                bg_gene_df_list.append(bg_gene_df)

        bg_gene_df_merge = pd.concat(bg_gene_df_list)
        bg_gene_df_merge.to_csv(bg_output_path, index=False)

    if remove_raw:
        for bg_fl in bg_fls:
            os.remove(bg_fl)

    return bg_gene_df_merge


# =============================================================================
# Model Checkpoint Utilities
# =============================================================================


def transfer_ft_indep_dirs(ft_indep_path: str) -> None:
    """
    Transfer independent fold results to standard cross-validation format.

    Args:
        ft_indep_path: Path to independent fold results directory.

    Note:
        Reorganizes fold directories and merges evaluation metrics.
    """
    if ft_indep_path[-1:] != "/":
        ft_indep_path = ft_indep_path + "/"

    fold_dirs = sorted([x for x in os.listdir(ft_indep_path) if "fold" in x])
    ft_indep_cv_path = os.path.join(ft_indep_path, "geneformer_cellRegressor_cv/")
    os.makedirs(ft_indep_cv_path, exist_ok=True)
    eval_metrics_dict_merge = {"mse": [], "mae": [], "r2": []}

    for fold_dir in fold_dirs:
        cv_dir = "ksplit" + str(int(fold_dir[-1:]) + 1)
        src_dir = os.path.join(ft_indep_path, fold_dir, "ksplit1")
        dest_dir = os.path.join(ft_indep_cv_path, cv_dir)
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

        eval_dict_path = os.path.join(
            ft_indep_path, fold_dir, f"{fold_dir.split('_')[-1]}_eval_metrics_dict.pkl"
        )
        with open(eval_dict_path, "rb") as f:
            eval_metrics_dict = pickle.load(f)

        for k, v in eval_metrics_dict.items():
            eval_metrics_dict_merge[k].append(v[0])

        shutil.rmtree(os.path.join(ft_indep_path, fold_dir))

        shutil.move(
            os.path.join(ft_indep_cv_path, cv_dir, f"fold{fold_dir[-1:]}_pred_dict.pkl"),
            os.path.join(ft_indep_cv_path, cv_dir, "cv_pred_dict.pkl"),
        )

    with open(os.path.join(ft_indep_cv_path, "cv_eval_metrics_dict.pkl"), "wb") as f:
        pickle.dump(eval_metrics_dict_merge, f)


def generate_pr_models_dict(
    path_checkpoint: str = "checkpoint_xe",
    work_dir: str = "/dataSSD7T/liss/work/scPCa/model/",
) -> Dict[str, List[str]]:
    """
    Generate dictionary of pretraining model checkpoints.

    Args:
        path_checkpoint: Path to checkpoint directory (relative to work_dir).
        work_dir: Working directory path.

    Returns:
        Dictionary mapping model IDs to [checkpoint_path, version].

    Example:
        >>> models_dict = generate_pr_models_dict()
        >>> gf_cl_path = models_dict["GF_CL"][0]
    """

    def filter_ck_by_pattern(path_checkpoint_xe: str) -> List[str]:
        pattern = re.compile(r"^D\d{4}_\d{2}")
        return [
            d
            for d in os.listdir(path_checkpoint_xe)
            if pattern.match(d) and os.path.isdir(os.path.join(path_checkpoint_xe, d))
        ]

    MODELS_DICT = {
        "GF_PR": [os.path.join(work_dir, "data/model/geneformer/gf-12L-95M-i4096"), "v3"],
        "GF_CL": [
            os.path.join(work_dir, "data/model/geneformer/gf-12L-95M-i4096_CLcancer"),
            "v3",
        ],
    }

    checkpoint_dirs = sorted(
        filter_ck_by_pattern(os.path.join(work_dir, path_checkpoint))
    )

    for checkpoint_dir in checkpoint_dirs:
        version = re.findall(r"_(v[2-9])_", checkpoint_dir)[0]
        model_id = "GF_" + re.findall(r"^D\d{4}_\d{2}", checkpoint_dir)[0]

        if model_id in MODELS_DICT:
            raise ValueError(f"Model ID {model_id} duplicated. Please check.")

        MODELS_DICT[model_id] = [
            os.path.join(work_dir, path_checkpoint, checkpoint_dir, "models"),
            version,
        ]

    return MODELS_DICT


# =============================================================================
# TensorBoard Log Utilities
# =============================================================================


def read_tensorboard_log(logdir: str) -> pd.DataFrame:
    """
    Read and parse TensorBoard event file.

    Args:
        logdir: Path to TensorBoard log directory.

    Returns:
        DataFrame with columns: tag, step, value, wall_time.

    Example:
        >>> df = read_tensorboard_log("./log/runs/exp1")
        >>> df[df["tag"] == "loss"]["value"].iloc[-1]
    """
    ea = event_accumulator.EventAccumulator(
        logdir, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    rows = []
    for tag in ea.Tags()["scalars"]:
        for event in ea.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                }
            )

    df = pd.DataFrame(rows)
    return df


def get_pr_model_log(
    model_id: str,
    ckpt_path: str = "./checkpoint_xe/",
    tf_log_path: str = "./log/runs/",
) -> pd.DataFrame:
    """
    Get TensorBoard log for a pretraining model, extracting max step for each tag.

    Args:
        model_id: Model identifier (e.g., "GF_D1120_12").
        ckpt_path: Path to checkpoint directory.
        tf_log_path: Path to TensorBoard log directory.

    Returns:
        DataFrame with max step entries for each tag, plus model_id column.

    Raises:
        FileNotFoundError: If checkpoint or log path not found.
        KeyError: If model_id not found.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    model_names = [m for m in os.listdir(ckpt_path) if m.startswith("D")]
    model_dict = {"_".join(m.split("_")[:2]): m for m in model_names}

    if model_id not in model_dict:
        raise KeyError(f"Model ID '{model_id}' not found in {ckpt_path}")

    sub_folder = f"PR_{model_id.split('_')[0]}"
    tf_log_model_path = os.path.join(tf_log_path, sub_folder, model_dict[model_id])

    if not os.path.exists(tf_log_model_path):
        raise FileNotFoundError(f"Log path does not exist: {tf_log_model_path}")

    log_df = read_tensorboard_log(tf_log_model_path)
    log_df = log_df.loc[log_df.groupby("tag")["step"].idxmax()].reset_index(drop=True)
    log_df["model_id"] = model_id

    return log_df


def get_ft_model_log(
    model_id: str,
    proj: str,
    set_id: str,
    ft_model_json_file: str,
    ft_task_dir: str = "./isp_gene_exp/",
) -> pd.DataFrame:
    """
    Get TensorBoard log for a fine-tuning model, extracting max step for each tag.

    Args:
        model_id: Model identifier.
        proj: Project name.
        set_id: Set identifier.
        ft_model_json_file: Path to model ID info JSON file.
        ft_task_dir: Fine-tuning task directory.

    Returns:
        DataFrame with max step entries for each tag, plus model metadata columns.

    Raises:
        FileNotFoundError: If required files or directories not found.
        KeyError: If model_id not found in JSON file.
    """
    with open(ft_model_json_file, "r") as f:
        ft_model_id_info = json.load(f)

    ft_model_id = {v: k for k, v in ft_model_id_info[proj][set_id].items()}[model_id]

    train_all_path = os.path.join(
        ft_task_dir,
        f"{proj}/output_ft/{set_id}/{ft_model_id}/train_all/geneformer_cellRegressor_all/",
    )

    if not os.path.exists(train_all_path):
        raise FileNotFoundError(f"Training output path not found: {train_all_path}")

    logs = os.listdir(os.path.join(train_all_path, "runs/"))
    if len(logs) != 1:
        raise ValueError(f"Expected 1 log directory, found {len(logs)}")

    log_df = read_tensorboard_log(os.path.join(train_all_path, "runs/", logs[0]))
    log_df = log_df.loc[log_df.groupby("tag")["step"].idxmax()].reset_index(drop=True)
    log_df["model_id"] = model_id
    log_df["proj"] = proj
    log_df["set_id"] = set_id
    log_df["ft_model_id"] = ft_model_id

    return log_df