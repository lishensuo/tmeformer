"""
TMEformer ISP Gene Expression Utilities
=======================================

This module provides utilities for in silico perturbation (ISP) gene expression
analysis, including:
- Cross-validation evaluation metrics
- Model visualization
- Background gene list preparation
- ISP gene rank analysis
- ISP score calculation and statistics

Key Functions:
    - cv_eval_metrics_df: Cross-validation evaluation metrics
    - calc_best_cv_ray: Calculate best CV ray
    - vis_gep_model_cv_boxplot: Visualize CV metrics
    - prep_bg_gene_lists: Prepare background gene lists
    - prep_isp_gene_rank_in_cells: Record ISP gene rank
    - merge_isp_gep_stat_raw: Merge ISP statistics
    - filter_isp_gep_stat_raw: Filter ISP statistics
    - calc_gep_isp_score_from_cell: Calculate ISP score for one cell
    - calc_gep_cell_isp_score_from_cells: Calculate ISP scores for cells
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, load_from_disk
from tqdm import tqdm

from . import TmeModeling_utils as tu
from . import TmeModeling_utils_isp_cell as tu_isp_cell
from . import TmeModeling_utils_isp_ds as tu_isp_ds
from . import TmeModeling_utils_isp_lst as tu_isp_lst


# =============================================================================
# Cross-Validation Evaluation Utilities
# =============================================================================


def cv_eval_metrics_df(
    proj: str,
    set_id: str,
    ft_model_id: str,
    ray_id: str,
    work_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarize the cross-validation evaluation metrics for fine-tune models.

    Args:
        proj: Project name.
        set_id: Set identifier.
        ft_model_id: Fine-tune model identifier.
        ray_id: Ray identifier.
        work_dir: Working directory path. If None, uses current directory.

    Returns:
        Combined evaluation metrics DataFrame with columns:
            fold, gene, mse, mae, r2, R, ft_model_id.
    """
    if work_dir is None:
        work_dir = "./"

    ft_dir = f"isp_gene_exp/{proj}/output_ft/{set_id}/{ft_model_id}"
    ray_dir = work_dir + f"{ft_dir}/{ray_id}/geneformer_cellRegressor_cv"

    with open(ray_dir + "/cv_eval_metrics_dict.pkl", "rb") as f:
        cv_eval_metrics_dict = pickle.load(f)

    with open(work_dir + f"{ft_dir}/gene_set.pickle", "rb") as f:
        gene_set = pickle.load(f)

    folds_gene_cors = []
    n_folds = len([x for x in os.listdir(ray_dir) if "ksplit" in x])

    for fold_idx in range(1, n_folds + 1):
        pk_path = f"{ray_dir}/ksplit{fold_idx}/cv_pred_dict.pkl"

        with open(pk_path, "rb") as f:
            pred_dict = pickle.load(f)

        pred_df = pd.DataFrame(
            pred_dict["pred_ids"],
            columns=[f"pred_{gene}" for gene in gene_set],
        )
        label_df = pd.DataFrame(
            pred_dict["label_ids"],
            columns=[f"label_{gene}" for gene in gene_set],
        )
        df_merge = pd.merge(pred_df, label_df, left_index=True, right_index=True)

        fold_gene_cors = []
        for gene in gene_set:
            pred_col = f"pred_{gene}"
            label_col = f"label_{gene}"
            cor_val = (
                df_merge[[pred_col, label_col]]
                .query(f"{label_col} != -1.0")
                .corr(method="pearson")
                .iloc[0, 1]
            )
            fold_gene_cors.append(cor_val)
        folds_gene_cors.append(fold_gene_cors)

    cv_eval_metrics_dict["R"] = folds_gene_cors

    dfs = []
    for metric in ["mse", "mae", "r2", "R"]:
        df = pd.DataFrame(cv_eval_metrics_dict[metric], columns=gene_set)
        df["fold"] = [f"fold{i}" for i in df.index]
        df["metric"] = metric
        dfs.append(df)

    combined_df = pd.concat(dfs).reset_index(drop=True)
    combined_df = (
        combined_df.melt(id_vars=["fold", "metric"], var_name="gene", value_name="value")
        .pivot(index=["fold", "gene"], columns="metric", values="value")
        .reset_index()
    )
    combined_df.columns.name = None
    combined_df["ft_model_id"] = ft_model_id

    return combined_df


def calc_best_cv_ray(
    eval_df_merge: pd.DataFrame,
    metric: str = "mse",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calculate the best cross-validation ray for each model.

    Args:
        eval_df_merge: Merged evaluation DataFrame.
        metric: Metric to use for selection. Options: 'mse', 'mae', 'r2', or 'R'.

    Returns:
        Tuple containing:
            - best_eval_df: Best evaluation DataFrame.
            - best_eval_df_mean: Mean metric values.
            - best_eval_df_median: Median metric values.

    Raises:
        ValueError: If metric is not one of the supported options.
    """
    avg_metric = (
        eval_df_merge.groupby(["model_id", "gene"], observed=True)[metric]
        .mean()
        .reset_index()
    )

    if metric in ["mse", "mae"]:
        best_avg_metric = (
            avg_metric.loc[
                avg_metric.groupby(["model_id", "gene"], observed=True)[metric].idxmin()
            ]
            .reset_index(drop=True)
        )
    elif metric in ["r2", "R"]:
        best_avg_metric = (
            avg_metric.loc[
                avg_metric.groupby(["model_id", "gene"], observed=True)[metric].idxmax()
            ]
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    best_eval_df = pd.merge(
        eval_df_merge,
        best_avg_metric[["model_id", "gene"]],
        on=["model_id", "gene"],
    )

    best_eval_df_mean = (
        best_eval_df.groupby(["model_id", "ft_model_id", "gene"], observed=True)[
            metric
        ].mean()
    )
    best_eval_df_median = (
        best_eval_df.groupby(["model_id", "ft_model_id", "gene"], observed=True)[
            metric
        ].median()
    )

    return best_eval_df, best_eval_df_mean, best_eval_df_median


# =============================================================================
# Visualization Utilities
# =============================================================================


def vis_gep_model_cv_boxplot(
    best_eval_df: pd.DataFrame,
    metric: str = "mse",
    gene: Optional[str] = None,
    title: Optional[str] = None,
    model_color_map: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None:
    """
    Visualize the CV metrics for models (boxplot).

    Args:
        best_eval_df: Best evaluation DataFrame.
        metric: Metric to plot (default: 'mse').
        gene: Specific gene to plot. None for all genes.
        title: Plot title.
        model_color_map: Color mapping for models.
        **kwargs: Additional arguments passed to sns.catplot.

    Note:
        If gene is not specified, plots a facet plot for all genes.
        Mean values are shown as red scatter points.
    """
    # If not gene specified, plot facet plot for all genes.
    if gene is not None:
        plot_data = best_eval_df[best_eval_df["gene"] == gene]
    else:
        plot_data = best_eval_df.copy()

    g = sns.catplot(
        data=plot_data,
        x="model_id",
        y=metric,
        hue="model_id",
        row="gene",  # one row per gene
        sharey=False,  # ylim not shared
        palette=model_color_map,
        **kwargs,
    )

    # Add the mean value points
    for ax, (gene_name, subdata) in zip(g.axes.flatten(), plot_data.groupby("gene")):
        mean_df = (
            subdata.groupby("model_id", as_index=False, observed=True)[metric].mean()
        )
        sns.scatterplot(
            data=mean_df,
            x="model_id",
            y=metric,
            color="red",
            s=50,
            zorder=10,
            marker="o",
            ax=ax,
            label="Mean",
            legend=False,
        )

        ax.set_title(f"Gene: {gene_name}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, size=1, labelsize=10)
        plt.setp(ax.get_xticklabels(), ha="right")

        for tick_label in ax.get_xticklabels():
            model_name = tick_label.get_text()
            color = model_color_map.get(model_name, "black")
            tick_label.set_color(color)

    if g._legend is not None:
        g._legend.remove()

    if title is not None:
        g.figure.suptitle(title, fontsize=16)
        g.figure.subplots_adjust(top=0.92)

    plt.show()


# =============================================================================
# Background Gene List Utilities
# =============================================================================


def prep_bg_gene_lists(
    proj: str = "xenium",
    set_id: str = "SET1",
    num_genes_per_isp: int = 1,
    num_isp: int = 100,
    pred_cells: str = "internal",
    work_dir: Optional[str] = None,
    existed: bool = True,
) -> Optional[str]:
    """
    Prepare background gene lists for ISP to calculate the distribution of
    background perturbation.

    Args:
        proj: Project name (default: 'xenium').
        set_id: Set identifier (default: 'SET1').
        num_genes_per_isp: Number of genes per ISP (default: 1).
        num_isp: Number of ISP samples (default: 100).
        pred_cells: "internal" or names of external project (default: 'internal').
        work_dir: Working directory path.
        existed: Whether to check for existing files (default: True).

    Returns:
        Path to saved file, or None if file already exists.

    Note:
        For internal datasets, samples genes from expressed set.
        For external datasets, samples genes randomly.
    """
    if work_dir is None:
        work_dir = "./"

    # Internal datasets, sampling gene(s) expressed set genes (not necessary)
    if pred_cells == "internal":
        suffix = ""
        genes_valid = tu.get_all_valid_genes(proj, work_dir=work_dir)
    # External datasets (pred_cells is proj name), sampling gene(s) randomly
    else:
        suffix = f"_{pred_cells}"
        genes_valid = tu.get_all_valid_genes(pred_cells, work_dir=work_dir)

    # check if already exists
    save_name = f"random_{num_genes_per_isp}genes_bg_cells_Model_{set_id}{suffix}.dict"
    if existed is False:
        save_name = save_name.replace(".dict", "_ki.dict")
    save_path = work_dir + f"isp_gene_exp/{proj}/datasets/background/{save_name}"
    if os.path.exists(save_path):
        print(f"{save_path} already exists")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 10 times of num_isp in case datasets not have enough cells that expressed this gene
    np.random.seed(42)
    genes_valid_bg = np.array(
        [
            np.random.choice(genes_valid, size=num_genes_per_isp, replace=False)
            for _ in range(num_isp * 10)
        ]
    ).tolist()

    dataset_path = work_dir + f"isp_gene_exp/{proj}/datasets/isp_vanilla_{set_id}{suffix}.dataset"

    if not os.path.exists(dataset_path):
        # if not have isp_vanilla dataset, use isp_v3 to generate
        dataset_tme = load_from_disk(dataset_path.replace("isp_vanilla", "isp_v3"))
        dataset_tme = dataset_tme.select_columns(
            ["input_ids", "cell_id", "sample_id", "length", "label"]
        )
        dataset_tme.save_to_disk(dataset_path)

    dataset = load_from_disk(dataset_path)

    genes_valid_bg_dict = {}

    def express_or_not(example: Dict[str, Any], isp_gene_token: List[int]) -> Dict[str, bool]:
        return {"expressed": set(isp_gene_token).issubset(example["input_ids"])}

    # Filter cells that expressed isp gene (>=10)
    for isp_gene_list in genes_valid_bg:
        isp_gene_token = [tu.symbol2token(gene_symbol) for gene_symbol in isp_gene_list]
        dataset_filt = dataset.map(
            express_or_not,
            num_proc=20,
            fn_kwargs={"isp_gene_token": isp_gene_token},
        )
        if existed is True:
            # cell expressed isp gene
            isp_valid_cells = np.where(np.array(dataset_filt["expressed"]))[0]
        else:
            # cell not expressed isp gene
            isp_valid_cells = np.where(~np.array(dataset_filt["expressed"]))[0]

        if isp_valid_cells.shape[0] < 10:
            continue

        genes_valid_bg_dict[tuple(isp_gene_list)] = isp_valid_cells
        n_dict = len(genes_valid_bg_dict)
        print(f"Now {n_dict}/{num_isp} done")

        if n_dict == num_isp:
            break

    with open(save_path, "wb") as f:
        pickle.dump(genes_valid_bg_dict, f)

    print("Done! ", save_path)
    return save_path


# =============================================================================
# ISP Gene Rank Utilities
# =============================================================================


def prep_isp_gene_rank_in_cells(
    proj: str = "xenium",
    base_model: str = "GF_PR",
    isp_run_name: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Record the pct expression interval of isp_genes in perturbed cell dataset.

    Args:
        proj: Project name (default: 'xenium').
        base_model: Base model identifier (default: 'GF_PR').
        isp_run_name: ISP run name.
        work_dir: Working directory path.

    Returns:
        Path to saved file, or None if file already exists.

    Example:
        >>> isp_run_name = "SET1_random-Target_Rank-S5000-W100_EP1_KO0_KI0-AHR"
    """
    if work_dir is None:
        work_dir = "./"

    save_path = work_dir + f"isp_gene_exp/{proj}/datasets/interval/{isp_run_name}.pickle"

    if os.path.exists(save_path):
        print(f"{save_path} exists")
        return None

    name_parts = isp_run_name.split("-", 4)
    # e.g. isp_run_name = "SET1_random-Target_Rank-S5000-W100_EP1_KO0_KI0-AHR"
    # → ['SET1_random', 'Target_Rank', 'S5000', 'W100_EP1_KO0_KI0', 'AHR']
    if "_" not in name_parts[0]:
        set_id = name_parts[0]
        ds_file_suffix = ""
    else:
        set_id, ds_file_suffix = name_parts[0].split("_")
        ds_file_suffix = "_" + ds_file_suffix

    gene_symbol = name_parts[-1]
    isp_type = name_parts[1].lower()

    isp_output = pd.read_csv(
        work_dir
        + f"isp_gene_exp/{proj}/output_isp/{base_model}/{set_id}/{isp_type}/{isp_run_name}.csv"
    )

    if "isp_gene" not in isp_output.columns:
        isp_output["isp_gene"] = gene_symbol

    isp_output_cells = (
        isp_output[["isp_gene", "cell_id"]]
        .drop_duplicates()
        .apply(lambda x: (x["isp_gene"], x["cell_id"]), axis=1)
        .values.tolist()
    )

    dataset = load_from_disk(
        work_dir
        + f"isp_gene_exp/{proj}/datasets/ft_vanilla_{set_id}_labeled{ds_file_suffix}.dataset"
    )
    dataset_cell_id = {cell_id: i for i, cell_id in enumerate(dataset["cell_id"])}

    rank_pc_list = []
    for isp_gene, isp_cell in tqdm(isp_output_cells):
        input_ids = dataset.select([dataset_cell_id[isp_cell]])[0]["input_ids"]
        x2 = len(input_ids) - 2

        if "+" not in isp_gene:  # single gene isp
            isp_gene_token = tu.symbol2token(isp_gene)
            x1 = input_ids.index(isp_gene_token)
            rank_pc = x1 / x2
            rank_pc_list.append(rank_pc)
        else:  # gene combination isp
            isp_gene_tokens = [tu.symbol2token(gene) for gene in isp_gene.split("+")]
            xx1 = [input_ids.index(isp_gene_token) for isp_gene_token in isp_gene_tokens]
            rank_pcs = [x1 / x2 for x1 in xx1]
            rank_pc_min_max = (min(rank_pcs), max(rank_pcs))
            rank_pc_list.append(rank_pc_min_max)

    rank_pc_dict = dict(zip(isp_output_cells, rank_pc_list))

    with open(save_path, "wb") as f:
        pickle.dump(rank_pc_dict, f)

    return save_path


# =============================================================================
# ISP Statistics Utilities
# =============================================================================


def merge_isp_gep_stat_raw(
    proj: str,
    conditions_dict: Dict[str, List[str]],
    model_ids: List[str],
    add_interval: bool = True,
    tme_isp: bool = False,
    tme_method: str = "composition",
    work_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge ISP GEP results of grouping conditions.

    Args:
        proj: Project name.
        conditions_dict: Dictionary of conditions, e.g.:
            {
                "Target": ["SET1_random-Target_Rank-S5000-W100_EP1_KO0_KI0-SOX2"],
                "Background": ["SET1_random-Target_Rank-S10000-W100_EP1_KO0_KI0-Background_C1"]
            }
        model_ids: List of model identifiers.
        add_interval: Whether to add interval information (default: True).
        tme_isp: Whether this is TME ISP (default: False).
        tme_method: TME method. Options: 'composition' or 'rank' (default: 'composition').
        work_dir: Working directory path.

    Returns:
        Merged ISP statistics DataFrame.

    Raises:
        FileNotFoundError: If interval file does not exist when add_interval=True.
        ValueError: If merge results in unexpected row count.
    """
    if work_dir is None:
        work_dir = "./"

    isp_score_merge_compare = []

    for k, v_list in conditions_dict.items():
        for condition in v_list:
            condi_parts = condition.split("-", 4)
            # ['SET1_random', 'Target_Rank', 'S5000', 'W100_EP1_KO0_KI0', 'ADNP']
            set_id = condi_parts[0].split("_")[0]
            isp_type = condi_parts[1].lower()
            isp_config = condi_parts[-2]
            perturb_obj = condi_parts[-1]

            isp_score_merge = []
            # Merge each condition result across models
            for model_id in model_ids:
                data_path = (
                    work_dir
                    + f"isp_gene_exp/{proj}/output_isp/{model_id}/{set_id}/{isp_type}/{condition}.csv"
                )

                if not os.path.exists(data_path):
                    print(f"[Warning] File not exists: {data_path}")
                    continue

                isp_score = pd.read_csv(data_path, index_col=0)
                isp_score.insert(0, "model_id", model_id)
                isp_score["pred_bias_pc"] = isp_score.apply(
                    lambda x: abs(x["exp_pred"] - x["exp_true"]), axis=1
                )
                isp_score = isp_score.rename(columns={"isp_score": "cell_score"})
                isp_score_merge.append(isp_score)

            if len(isp_score_merge) == 0:
                continue

            isp_score_merge = pd.concat(isp_score_merge)

            # Add additional columns for different isp types
            if tme_isp:
                celltype = int(perturb_obj.split("_")[0].replace("TME", ""))
                isp_score_merge["cell_type"] = celltype
                isp_score_merge["group"] = celltype if k == "Target" else "Background"

                if tme_method == "composition":
                    fold_change = float(isp_config.split("_")[1].replace("EP", ""))
                    isp_score_merge["fold_change"] = fold_change
                    isp_score_merge = isp_score_merge.rename(
                        columns={"cell_type": "isp_cell_type"}
                    )
                elif tme_method == "rank":
                    gene_symbol = perturb_obj.split("_")[1]
                    isp_score_merge["isp_gene"] = gene_symbol
            else:
                isp_gene_type = perturb_obj
                isp_score_merge["group"] = isp_gene_type
                if "Background" in isp_gene_type:
                    isp_score_merge = isp_score_merge.rename(
                        columns={"gene": "isp_gene"}
                    )
                else:
                    isp_score_merge["isp_gene"] = isp_gene_type

            isp_score_merge_compare.append(isp_score_merge)

    isp_score_merge_compare = pd.concat(isp_score_merge_compare)
    isp_score_merge_compare["model_id"] = pd.Categorical(
        isp_score_merge_compare["model_id"],
        categories=model_ids,
        ordered=True,
    )

    # Add pct expression interval of isp genes
    if add_interval:
        interval_df_list = []
        for k, v_list in conditions_dict.items():
            for condition in v_list:
                interval_path = (
                    work_dir + f"isp_gene_exp/{proj}/datasets/interval/{condition}.pickle"
                )

                if not os.path.exists(interval_path):
                    raise FileNotFoundError(
                        f"{interval_path} does not exist. Prepare it by "
                        "tu_isp_gep.prep_isp_gene_rank_in_cells() function."
                    )

                with open(interval_path, "rb") as f:
                    interval_dict = pickle.load(f)

                interval_df = pd.DataFrame(
                    [(gene, cell_id, score) for (gene, cell_id), score in interval_dict.items()],
                    columns=["isp_gene", "cell_id", "rank_pc"],
                )
                interval_df["group"] = condition.split("-", 4)[-1]
                interval_df_list.append(interval_df)

        interval_df_merge = pd.concat(interval_df_list)

        n_row = isp_score_merge_compare.shape[0]
        isp_score_merge_compare = pd.merge(isp_score_merge_compare, interval_df_merge)

        if isp_score_merge_compare.shape[0] != n_row:
            x1 = isp_score_merge_compare.shape[0]
            x2 = n_row
            raise ValueError(f"Merge failed ({x1} != {x2}). Please check")

    return isp_score_merge_compare


def filter_isp_gep_stat_raw(
    stats_raw: pd.DataFrame,
    score_type: Optional[str] = "OE",
    score_method: Optional[str] = "endpoint",
    score_interval: Optional[str] = None,
    stat_level: str = "patch",
    stat_method: str = "mean",
    pair: bool = True,
) -> pd.DataFrame:
    """
    Filter and aggregate ISP GEP results.

    Args:
        stats_raw: Raw statistics DataFrame.
        score_type: Type of score. Options: 'OE', 'KD', or None.
        score_method: Method for scoring. Options: 'endpoint', 'area', or None.
        score_interval: Score interval to filter. None for no filtering.
        stat_level: Level of statistics. Options: 'patch' or 'sample'.
        stat_method: Statistical method to apply (default: 'mean').
        pair: Whether to keep paired ISP between groups (default: True).

    Returns:
        Filtered statistics DataFrame.

    Raises:
        ValueError: If score_interval is specified but no rank_pc column exists.
    """
    stats_filt = stats_raw.copy()

    if score_type is not None:
        stats_filt = stats_filt[stats_filt["score_type"] == score_type]

    if score_method is not None:
        stats_filt = stats_filt[stats_filt["score_method"] == score_method]

    if score_interval is not None:
        if "rank_pc" in stats_filt.columns:
            stats_filt = add_interval_annotation(stats_filt, score_type)
            stats_filt = stats_filt[stats_filt["interval"] == score_interval]
        else:
            raise ValueError("[Error] No rank_pc column in stats_raw")

    if stat_level == "patch":
        default_group_cols = [
            "model_id",
            "model_gene",
            "patch_id",
            "score_type",
            "score_method",
        ]
        group_cols = (
            default_group_cols + ["interval"]
            if "interval" in stats_raw.columns
            else default_group_cols
        )

        stats_filt = (
            stats_filt.groupby(group_cols + ["group"], observed=True)["cell_score"]
            .agg([stat_method])
            .reset_index()
        )

        if stats_filt["group"].unique().shape[0] > 1:
            stats_filt = stats_filt.dropna()
            # Keep paired isp between two groups
            if pair:
                stats_filt = stats_filt[stats_filt[group_cols].duplicated(keep=False)]

    elif stat_level == "sample":
        default_group_cols = [
            "model_id",
            "model_gene",
            "sample_name",
            "score_type",
            "score_method",
        ]
        group_cols = (
            default_group_cols + ["interval"]
            if "interval" in stats_raw.columns
            else default_group_cols
        )

        stats_filt = (
            stats_filt.groupby(group_cols + ["group"], observed=True)["cell_score"]
            .agg([stat_method])
            .reset_index()
        )

    stats_filt = stats_filt.rename(columns={stat_method: "cell_score"})

    return stats_filt


def add_interval_annotation(
    stats_raw: pd.DataFrame,
    score_type: str,
) -> pd.DataFrame:
    """
    Add expression interval annotation of isp genes among perturbed cells.

    Args:
        stats_raw: Raw statistics DataFrame.
        score_type: Type of score. Options: 'OE' or 'KD'.

    Returns:
        Statistics DataFrame with interval annotations.

    Raises:
        ValueError: If score_type is not 'OE' or 'KD'.
    """
    groups = stats_raw["group"].unique().tolist()
    group_target = groups[0] if groups[1].startswith("Background") else groups[1]
    comb_isp = "+" in group_target

    if score_type == "OE":
        intervals = [[0, 1], [0.5, 1], [0.75, 1]]
    elif score_type == "KD":
        intervals = [[0, 1], [0, 0.5], [0, 0.25]]
    else:
        raise ValueError(f"Invalid score_type: {score_type}")

    models_isp_score_intervals = []

    for interval in intervals:
        if score_type == "OE":
            rank_func = (
                (lambda x: x >= interval[0]) if not comb_isp else (lambda x: x[0] >= interval[0])
            )
        elif score_type == "KD":
            rank_func = (
                (lambda x: x <= interval[1]) if not comb_isp else (lambda x: x[1] <= interval[1])
            )

        models_isp_score_interval = (
            stats_raw.loc[stats_raw["rank_pc"].apply(rank_func)].copy()
        )

        models_isp_score_interval["interval"] = str(interval)
        models_isp_score_intervals.append(models_isp_score_interval)

    models_isp_score_intervals = pd.concat(models_isp_score_intervals)

    return models_isp_score_intervals


# =============================================================================
# ISP Score Calculation Utilities
# =============================================================================


def calc_gep_isp_score_from_cell(
    isp_dict: Dict[int, float],
    isp_run_config: Dict[str, Any],
    isp: str = "cell",
) -> pd.DataFrame:
    """
    Calculate ISP score for one cell.

    Args:
        isp_dict: ISP dictionary with offset as key and score as value.
        isp_run_config: ISP run configuration.
        isp: ISP type. Options: 'cell', 'tme_composition', or 'tme_rank'.

    Returns:
        ISP scores DataFrame with columns: score_type, score_method, isp_score.
    """
    # Decide isp score sets (score type, score method)
    if isp == "cell":
        isp_score_sets = tu_isp_ds.generate_cell_isp_score_sets(isp_run_config)
    elif isp == "tme_composition":
        isp_score_sets = tu_isp_ds.generate_tme_isp_score_sets(
            isp_run_config, method="composition"
        )
    elif isp == "tme_rank":
        isp_score_sets = tu_isp_ds.generate_tme_isp_score_sets(
            isp_run_config, method="rank"
        )
    else:
        raise ValueError(f"Invalid isp type: {isp}")

    # Calculate isp scores
    isp_scores_dict = defaultdict(list)
    for score_type, score_method in isp_score_sets:
        isp_scores_dict["score_type"].append(score_type)
        isp_scores_dict["score_method"].append(score_method)
        isp_scores_dict["isp_score"].append(
            tu_isp_ds.calc_isp_score(isp_dict, score_type, score_method)
        )

    isp_scores_df = pd.DataFrame(isp_scores_dict)
    return isp_scores_df


def calc_gep_cell_isp_score_from_cells(
    dataset: Dataset,
    isp_gene_token: List[int],
    model: torch.nn.Module,
    isp_run_config: Dict[str, Any],
    batch: int = 16,
    pred_idx: int = 0,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Calculate ISP score (Target) for cells.

    Args:
        dataset: Dataset containing cells.
        isp_gene_token: ISP gene token list.
        model: Model for prediction.
        isp_run_config: ISP run configuration.
        batch: Batch size for prediction (default: 16).
        pred_idx: Prediction index (default: 0).
        logger: Logger object.

    Returns:
        Cell ISP scores DataFrame with columns:
            cell_id, exp_true, exp_pred, score_type, score_method, isp_score.
    """
    cells_isp_reg_score_df = []

    for i in range(len(dataset)):
        if (i + 1) % 500 == 0:
            if logger is not None:
                logger.info(f"==> Processing cell {i+1}/{len(dataset)}")

        example_cell = dataset.select([i])

        shifted_cell_dataset, _ = tu_isp_lst.perturb_one_cell(
            example_cell, isp_gene_token, model, isp_run_config, do_embed=False
        )

        output = tu.model_predict(
            model, shifted_cell_dataset, forward_batch_size=batch, verbose=False
        )

        isp_reg_dict = dict(
            zip(np.array(shifted_cell_dataset["offset"]), np.array(output[0])[:, pred_idx].squeeze())
        )
        isp_reg_dict_relative = {x: [y - isp_reg_dict[0]] for x, y in isp_reg_dict.items()}

        cell_isp_reg_score_df = calc_gep_isp_score_from_cell(
            isp_reg_dict_relative, isp_run_config
        )
        cell_isp_reg_score_df.insert(0, "cell_id", example_cell["cell_id"][0])

        if isinstance(example_cell["label"][0], list):
            cell_isp_reg_score_df.insert(1, "exp_true", example_cell["label"][0][pred_idx])
        else:
            cell_isp_reg_score_df.insert(1, "exp_true", example_cell["label"][0])

        cell_isp_reg_score_df.insert(2, "exp_pred", isp_reg_dict[0])

        cells_isp_reg_score_df.append(cell_isp_reg_score_df)

    cells_isp_reg_score_df = pd.concat(cells_isp_reg_score_df).reset_index(drop=True)

    return cells_isp_reg_score_df


def calc_gep_tme_isp_score_from_cells(
    dataset: Dataset,
    tme_method: str,
    isp_cluster_id: int,
    model: torch.nn.Module,
    cell_tme_dict: Dict[str, Any],
    cell_cluster_dict: Dict[str, Any],
    isp_run_config: Dict[str, Any],
    batch: int = 16,
    pred_idx: int = 0,
    logger: Optional[Any] = None,
    work_dir: Optional[str] = None,
    proj: str = "xenium",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Calculate ISP score (TME) for cells.

    Args:
        dataset: Dataset containing cells.
        tme_method: TME method. Options: 'composition' or 'rank'.
        isp_cluster_id: ISP cluster identifier.
        model: Model for prediction.
        cell_tme_dict: Cell TME dictionary.
        cell_cluster_dict: Cell cluster dictionary.
        isp_run_config: ISP run configuration.
        batch: Batch size for prediction (default: 16).
        pred_idx: Prediction index (default: 0).
        logger: Logger object.
        work_dir: Working directory path.
        proj: Project name (default: 'xenium').
        **kwargs: Additional arguments:
            - fixed_cluster_ids: Required for 'composition' method.
            - gene_symbol: Required for 'rank' method.
            - resample: Optional for 'rank' method (default: True).

    Returns:
        Cell ISP scores DataFrame.

    Raises:
        ValueError: If required parameters are missing for the specified tme_method.
    """
    if work_dir is None:
        work_dir = "./"

    if tme_method == "composition":
        for param in ["fixed_cluster_ids"]:
            if param not in kwargs:
                raise ValueError(f"{param} is required for tme_method 'composition'")
        fixed_cluster_ids = kwargs.get("fixed_cluster_ids", None)
    elif tme_method == "rank":
        for param in ["gene_symbol", "resample"]:
            if param not in kwargs:
                raise ValueError(f"{param} is required for tme_method 'rank'")
        gene_symbol = kwargs.get("gene_symbol", None)
        resample = kwargs.get("resample", True)
        gene_token = tu.symbol2token(gene_symbol)
    else:
        raise ValueError(f"Invalid tme_method: {tme_method}")

    cells_isp_reg_score_df = []

    for i in range(len(dataset)):
        if (i + 1) % 500 == 0:
            if logger is not None:
                logger.info(f"==> Processing cell {i+1}/{len(dataset)}")

        example_cell = dataset.select([i])

        if tme_method == "composition":
            shifted_cell_dataset, _ = tu_isp_cell.perturb_one_tme_cells_composition(
                example_cell,
                isp_cluster_id,
                fixed_cluster_ids,
                model,
                cell_tme_dict,
                cell_cluster_dict,
                isp_run_config,
            )
            output = tu.model_predict(
                model, shifted_cell_dataset, forward_batch_size=batch, verbose=False
            )
        elif tme_method == "rank":
            # Set emb_type to "tme" due to the model is fine-tuned not pretrained
            # model, so return is tme_cell_embs
            shifted_cell_dataset, tme_cell_embs = tu_isp_cell.perturb_one_tme_cells_rank(
                example_cell,
                isp_cluster_id,
                gene_token,
                model,
                cell_cluster_dict,
                isp_run_config,
                resample=resample,
                emb_type="tme",
                work_dir=work_dir,
                proj=proj,
            )
            if tme_cell_embs is None:
                # TME cells not express the isp gene, so cannot be perturbed
                continue
            output = tu.model_predict(
                model,
                shifted_cell_dataset,
                forward_batch_size=batch,
                verbose=False,
                tme_cell_embs=tme_cell_embs,
            )

        isp_reg_dict = dict(
            zip(np.array(shifted_cell_dataset["offset"]), np.array(output[0])[:, pred_idx].squeeze())
        )
        isp_reg_dict_relative = {x: [y - isp_reg_dict[0]] for x, y in isp_reg_dict.items()}

        cell_isp_reg_score_df = calc_gep_isp_score_from_cell(
            isp_reg_dict_relative, isp_run_config, isp="tme_" + tme_method
        )
        cell_isp_reg_score_df.insert(0, "cell_id", example_cell["cell_id"][0])

        if isinstance(example_cell["label"][0], list):
            cell_isp_reg_score_df.insert(1, "exp_true", example_cell["label"][0][pred_idx])
        else:
            cell_isp_reg_score_df.insert(1, "exp_true", example_cell["label"][0])

        cell_isp_reg_score_df.insert(2, "exp_pred", isp_reg_dict[0])

        cells_isp_reg_score_df.append(cell_isp_reg_score_df)

    if logger is not None:
        logger.info(f"==> {len(cells_isp_reg_score_df)} cells are processed successfully")

    cells_isp_reg_score_df = pd.concat(cells_isp_reg_score_df).reset_index(drop=True)

    return cells_isp_reg_score_df


# =============================================================================
# Summary Utilities
# =============================================================================


def summary_isp_gep_stat_files(
    task_dir: Optional[str] = None,
    set_id: Optional[str] = None,
    isp_type: str = "target_rank",
    pred_cells: str = "internal",
    model_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Summary of ISP conditions for one set_id across models.

    Args:
        task_dir: Task directory path.
        set_id: Set identifier.
        isp_type: ISP type (default: 'target_rank').
        pred_cells: "internal" or names of external project (default: 'internal').
        model_ids: List of model identifiers to filter.

    Returns:
        Tuple containing:
            - isp_dict: Dictionary mapping conditions to model IDs.
            - condi_meta: DataFrame with condition metadata (perturb, config, condition).

    Raises:
        ValueError: If no expected task_stat_models found.
    """
    if task_dir is None:
        raise ValueError("task_dir must be provided")
    if set_id is None:
        raise ValueError("set_id must be provided")

    task_stat_dir = Path(task_dir)
    task_stat_models = [
        mid for mid in task_stat_dir.iterdir() if mid.name.startswith("GF")
    ]

    if model_ids is not None:
        task_stat_models = [mid for mid in task_stat_models if mid.name in model_ids]

    task_stat_models = [mid / set_id / isp_type for mid in task_stat_models]
    task_stat_models = [mid for mid in task_stat_models if mid.exists()]

    if len(task_stat_models) == 0:
        raise ValueError(f"No expected task_stat_models found in {task_stat_dir}")

    if pred_cells == "internal":
        expected_set_id_name = set_id
    else:
        expected_set_id_name = set_id + "_" + pred_cells

    isp_dict = defaultdict(list)

    for task_stat_model in task_stat_models:
        for file in task_stat_model.iterdir():
            set_full_name = file.stem.split("-")[0]

            if set_full_name == expected_set_id_name:
                isp_dict[file.stem].append(file.parent.parts[-3])

    isp_dict = dict(sorted(isp_dict.items()))

    conditions = list(isp_dict.keys())
    # e.g. "SET1_random-Target_Rank-S5000-W100_EP1_KO0_KI0-AHR"
    # → ['SET1_random', 'Target_Rank', 'S5000', 'W100_EP1_KO0_KI0', 'AHR']
    perturb = [condi.split("-", 4)[-1] for condi in conditions]
    config = [condi.split("-", 4)[-2] for condi in conditions]

    condi_meta = pd.DataFrame(
        {
            "perturb": perturb,
            "config": config,
            "condition": conditions,
        }
    )

    return isp_dict, condi_meta