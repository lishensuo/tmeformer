"""
TMEformer ISP Embedding Similarity Utilities
============================================

This module provides utilities for in silico perturbation (ISP) embedding
similarity analysis, including:
- Merging and filtering ISP similarity statistics
- Statistical analysis and p-value calculation
- Visualization of embedding similarity changes
- Summary of ISP similarity conditions

Key Functions:
    - merge_isp_sim_stat_raw: Merge ISP similarity statistics
    - filter_isp_sim_stat_raw: Filter and aggregate statistics
    - calc_summary_stat_pvals: Calculate summary statistics and p-values
    - one_sided_wilcoxon: Create one-sided Wilcoxon test
    - vis_delta_emb_sim_boxbar: Visualize delta embedding similarity
    - vis_pval_barplot: Visualize p-values
    - vis_adjusted_delta_violin: Visualize adjusted delta similarity
    - summary_isp_sim_stat_files: Summary of ISP conditions
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest


# =============================================================================
# ISP Similarity Statistics Utilities
# =============================================================================


def merge_isp_sim_stat_raw(
    conditions_dict: Dict[str, List[str]],
    task: str,
    model_ids: List[str],
    tme_isp: bool = False,
    tme_method: str = "composition",
    work_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge ISP similarity statistics from multiple conditions and models.

    Args:
        conditions_dict: Dictionary of conditions to process.
        task: Task identifier.
        model_ids: List of model identifiers.
        tme_isp: Whether to process TME ISP data (default: False).
        tme_method: Method for TME processing. Options: 'composition' or 'rank'.
        work_dir: Working directory path.

    Returns:
        DataFrame containing merged statistics.

    Raises:
        ValueError: If stat file does not exist.

    Example:
        >>> conditions_dict = {
        ...     "Target": ["ADT2CRPC-Target_Rank-L1-S1000-W30_EP1_KO0_KI0-ADNP"],
        ...     "Background": ["ADT2CRPC-Target_Rank-L1-S1000-W30_EP1_KO0_KI0-Background_C1"]
        ... }
    """
    if work_dir is None:
        work_dir = "./"

    stats_raw_list = []

    for k, v_list in conditions_dict.items():
        for condition in v_list:
            # e.g. condition = "ADT2CRPC_ADT>CRPC-Target_Rank-L1-S1000-W30_EP1_KO0_KI0-ADNP"
            # → ['ADT2CRPC_ADT>CRPC', 'Target_Rank', 'L1', 'S1000', 'W30_EP1_KO0_KI0', 'ADNP']
            condi_parts = condition.split("-", 5)
            isp_type = condi_parts[1].lower()
            isp_config = condi_parts[-2]
            perturb_obj = condi_parts[-1]

            stat_raw_paths = [
                work_dir + f"isp_emb_sim/task_{task}/stat/{model_id}/{isp_type}/{condition}.csv"
                for model_id in model_ids
            ]
            for stat_raw_path in stat_raw_paths:
                if not os.path.exists(stat_raw_path):
                    raise ValueError(f"Stat file not exists: {stat_raw_path}")

            stats_raw = pd.concat([pd.read_csv(raw_stat_path) for raw_stat_path in stat_raw_paths])
            stats_raw["model_id"] = pd.Categorical(stats_raw["model_id"], model_ids, ordered=True)

            if tme_isp:
                celltype = int(perturb_obj.split("_")[0].replace("TME", ""))
                stats_raw["cell_type"] = celltype
                stats_raw["group"] = celltype if k == "Target" else "Background"

                if tme_method == "composition":
                    fold_change = float(isp_config.split("_")[1].replace("EP", ""))
                    stats_raw["fold_change"] = fold_change
                elif tme_method == "rank":
                    gene_symbol = perturb_obj.split("_")[1]
                    stats_raw["gene_symbol"] = gene_symbol
                    stats_raw["group"] = gene_symbol if k == "Target" else "Background"
            else:
                isp_gene_type = perturb_obj
                stats_raw["group"] = isp_gene_type
                if "Background" in isp_gene_type:
                    stats_raw = stats_raw.rename(columns={"gene": "gene_symbol"})
                else:
                    stats_raw["gene_symbol"] = isp_gene_type

            stats_raw_list.append(stats_raw)

    stats_raw = pd.concat(stats_raw_list)
    return stats_raw


def filter_isp_sim_stat_raw(
    stats_raw: pd.DataFrame,
    score_type: Optional[str],
    score_method: Optional[str],
    score_interval: Optional[str] = None,
    stat_level: str = "patch",
    stat_method: str = "mean",
    pair: bool = True,
) -> pd.DataFrame:
    """
    Filter and aggregate ISP similarity statistics.

    Args:
        stats_raw: Raw statistics DataFrame.
        score_type: Type of score to filter.
        score_method: Method of score calculation.
        score_interval: Optional interval filter.
        stat_level: Level of statistics. Options: 'patch' or 'sample'.
        stat_method: Aggregation method (default: 'mean').
        pair: Whether to keep only paired data (default: True).

    Returns:
        Filtered and aggregated DataFrame.

    Example:
        >>> stats_filt = filter_isp_sim_stat_raw(stats_raw, "OE", "endpoint")
    """
    stats_filt = stats_raw.copy()

    if score_type is not None:
        stats_filt = stats_filt[stats_filt["score_type"] == score_type]
    if score_method is not None:
        stats_filt = stats_filt[stats_filt["score_method"] == score_method]

    if score_interval is not None and "interval" in stats_filt.columns:
        stats_filt = stats_filt[stats_filt["interval"] == score_interval]

    if stat_level == "patch":
        default_group_cols = ["model_id", "patch_id", "score_type", "score_method"]
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
        if stats_filt["group"].unique().shape[0] > 1:  # ["target", "background"]
            stats_filt = stats_filt.dropna()
            if pair:
                stats_filt = stats_filt[stats_filt[group_cols].duplicated(keep=False)]

    elif stat_level == "sample":
        default_group_cols = ["model_id", "sample_id", "score_type", "score_method"]
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


# =============================================================================
# Statistical Analysis Utilities
# =============================================================================


def calc_summary_stat_pvals(
    stats_filt: pd.DataFrame,
    stat_level: str,
    stat_method: str,
    wilcoxon_test: str = "less",
    result: str = "pval",
) -> pd.DataFrame:
    """
    Calculate summary statistics and p-values.

    Args:
        stats_filt: Filtered statistics DataFrame.
        stat_level: Level of statistics. Options: 'sample', 'patch', or 'cell'.
        stat_method: Statistical method to apply.
        wilcoxon_test: Alternative hypothesis for Wilcoxon test (default: 'less').
        result: Result type. Options: 'pval' or 'diff' (default: 'pval').

    Returns:
        DataFrame with summary statistics and p-values (or differences).

    Raises:
        ValueError: If stat_level is not one of the supported options.

    Example:
        >>> stat_summary = calc_summary_stat_pvals(stats_filt, "patch", "mean")
    """
    stat_level_col = {"sample": "sample_id", "patch": "patch_id"}.get(stat_level)

    groups_ordered = stats_filt["group"].unique().tolist()
    if "Background" not in groups_ordered[0]:
        groups_ordered = groups_ordered[::-1]

    if stat_level in ["sample", "patch"]:
        stats_filt = (
            stats_filt.groupby(["model_id", "group", stat_level_col], observed=True)[
                "cell_score"
            ]
            .agg(stat_method)
            .reset_index()
        )

    stat_data_summary = (
        stats_filt.groupby(["model_id", "group"], observed=False)
        .agg(stat_isp_score=("cell_score", stat_method))
        .reset_index()
        .pivot(index="model_id", columns="group", values="stat_isp_score")
    )
    stat_data_summary.columns.name = None
    stat_data_summary.index.name = None

    p_values = []
    data_diffs = []

    for model in stats_filt["model_id"].unique():
        if stat_level == "cell":
            group1 = stats_filt[
                (stats_filt["model_id"] == model)
                & (stats_filt["group"] == groups_ordered[0])
            ]["cell_score"].dropna()
            group2 = stats_filt[
                (stats_filt["model_id"] == model)
                & (stats_filt["group"] == groups_ordered[1])
            ]["cell_score"].dropna()
            # Mann-Whitney U test (wilcoxon rank sum test)
            _, p = mannwhitneyu(group1, group2, alternative="less")
            p_values.append({"model_id": model, "p_value": p})
        else:
            # Paired test (wilcoxon signed-rank test)
            stat_data_paired = stats_filt.query("model_id == @model")
            stat_data_paired = stat_data_paired[
                stat_data_paired[stat_level_col].duplicated(keep=False)
            ].sort_values(by=[stat_level_col])

            if len(stat_data_paired) == 0:
                p = np.nan
                data_diffs.append(
                    pd.DataFrame({"model_id": model, "cell_score_diff": [np.nan]})
                )

            else:
                group1_value = stat_data_paired[
                    stat_data_paired["group"] == groups_ordered[0]
                ][
                    "cell_score"
                ]  # "Background"
                group2_value = stat_data_paired[
                    stat_data_paired["group"] == groups_ordered[1]
                ][
                    "cell_score"
                ]  # "Target"
                # alternative='less' significance means group2_value > group1_value
                # alternative='greater' significance means group1_value > group2_value
                _, p = wilcoxon(group1_value, group2_value, alternative=wilcoxon_test)
                data_diffs.append(
                    pd.DataFrame(
                        {
                            "model_id": model,
                            "cell_score_diff": group2_value.values - group1_value.values,
                        }
                    )
                )

            p_values.append({"model_id": model, "p_value": p})

    if result == "pval":
        stat_data_summary = pd.merge(
            stat_data_summary, pd.DataFrame(p_values), left_index=True, right_on="model_id"
        )
        stat_data_summary = stat_data_summary.loc[
            :, ["model_id"] + groups_ordered + ["p_value"]
        ]

        stat_data_summary["log_p"] = -np.log10(stat_data_summary["p_value"])
        stat_data_summary["log_p_cut"] = np.where(
            stat_data_summary["log_p"] > 10, 10, stat_data_summary["log_p"]
        )
        return stat_data_summary

    if result == "diff":
        data_diffs = pd.concat(data_diffs)
        return data_diffs

    raise ValueError(f"Invalid result type: {result}")


def one_sided_wilcoxon(alternative: str = "less") -> StatTest:
    """
    Create a one-sided Wilcoxon test function.

    Args:
        alternative: Alternative hypothesis. Options: 'less' or 'greater'.

    Returns:
        StatTest object for one-sided Wilcoxon test.

    Example:
        >>> test = one_sided_wilcoxon("less")
    """

    def paired_wilcoxon_one_sided(
        group1: np.ndarray, group2: np.ndarray, **kwargs: Any
    ) -> Tuple[float, float]:
        g1 = np.asarray(group1)
        g2 = np.asarray(group2)

        # Drop NaN values and keep paired
        mask = ~np.isnan(g1) & ~np.isnan(g2)
        g1_clean = g1[mask]
        g2_clean = g2[mask]

        stat, pval = wilcoxon(g1_clean, g2_clean, **kwargs)
        return stat, pval

    custom_test = StatTest(
        func=paired_wilcoxon_one_sided,
        test_long_name="Paired Wilcoxon test (one-sided)",
        test_short_name="wilcoxon-one-sided",
        alternative=alternative,
    )

    return custom_test


# =============================================================================
# Visualization Utilities
# =============================================================================


def vis_delta_emb_sim_boxbar(
    stats_filt: pd.DataFrame,
    model_color_map: Optional[Dict[str, str]] = None,
    title_text: Optional[str] = None,
    xlabel_text: Optional[str] = None,
    ylabel_text: str = "Delta Embedding similarity",
    figsize: Tuple[int, int] = (8, 6),
    wilcoxon_test: Optional[str] = "less",
    geom_type: str = "box",
    new_model_names: Optional[List[str]] = None,
    lg_params: Optional[Dict[str, Any]] = None,
    line: bool = False,
    line_params: Optional[Dict[str, Any]] = None,
    pair_id_col: Optional[str] = None,
    xtick_rotation = 45,
    xtick_ha = "right",
    **kwargs: Any,
) -> None:
    """
    Visualize the distributions of cell ISP scores across given groups.

    Args:
        stats_filt: Filtered statistics DataFrame.
        model_color_map: Dictionary mapping model names to colors.
        title_text: Plot title.
        xlabel_text: X-axis label.
        ylabel_text: Y-axis label (default: 'Delta Embedding similarity').
        figsize: Figure size tuple (default: (8, 6)).
        wilcoxon_test: Alternative hypothesis for Wilcoxon test. None for no test.
        geom_type: Type of plot. Options: 'box' or 'bar'.
        new_model_names: List of new model names for x-axis.
        lg_params: Legend parameters dictionary.
        line: Whether to draw lines connecting paired samples (default: False).
        line_params: Line parameters dictionary.
        pair_id_col: Column name for pair IDs (required when line=True).
        **kwargs: Additional arguments passed to seaborn plot functions.

    Raises:
        ValueError: If pair_id_col is not specified when line=True.

    Example:
        >>> vis_delta_emb_sim_boxbar(stats_filt, x="model_id", hue="group")
    """
    plt.figure(figsize=figsize)

    if geom_type == "box":
        ax = sns.boxplot(data=stats_filt, y="cell_score", **kwargs)
    elif geom_type == "bar":
        ax = sns.barplot(data=stats_filt, y="cell_score", **kwargs)

    # Draw lines connecting paired samples
    if line:
        if pair_id_col is None:
            raise ValueError("pair_id_col must be specified when line=True")

        # Default line parameters
        default_line_params = {
            "color": "gray",
            "alpha": 0.3,
            "linewidth": 0.8,
            "zorder": 0,
        }
        if line_params is not None:
            default_line_params.update(line_params)

        # Get x-axis configuration
        x_col = kwargs.get("x")
        order = kwargs.get("order")
        hue_col = kwargs.get("hue")
        hue_order = kwargs.get("hue_order")

        # Get unique pair IDs
        pair_ids = stats_filt[pair_id_col].unique()

        # For each pair, draw connecting lines
        for pair_id in pair_ids:
            pair_data = stats_filt[stats_filt[pair_id_col] == pair_id]

            if hue_col is not None and hue_order is not None:
                # Paired by hue within each x category
                for x_val in order:
                    x_pair_data = pair_data[pair_data[x_col] == x_val]
                    if len(x_pair_data) != len(hue_order):
                        continue

                    # Get x positions for this group
                    x_idx = order.index(x_val)
                    n_hues = len(hue_order)
                    width = 0.8 / n_hues
                    positions = [
                        x_idx + width * (i - n_hues / 2 + 0.5) for i in range(n_hues)
                    ]

                    # Draw lines between hue levels
                    for i in range(len(hue_order) - 1):
                        hue_val1 = hue_order[i]
                        hue_val2 = hue_order[i + 1]

                        y1 = x_pair_data[x_pair_data[hue_col] == hue_val1][
                            "cell_score"
                        ].values
                        y2 = x_pair_data[x_pair_data[hue_col] == hue_val2][
                            "cell_score"
                        ].values

                        if len(y1) > 0 and len(y2) > 0:
                            ax.plot(
                                [positions[i], positions[i + 1]],
                                [y1[0], y2[0]],
                                **default_line_params,
                            )
            else:
                # Paired across x categories (no hue)
                if len(pair_data) != len(order):
                    continue

                x_positions = []
                y_values = []

                for x_val in order:
                    x_data = pair_data[pair_data[x_col] == x_val]
                    if len(x_data) > 0:
                        x_positions.append(order.index(x_val))
                        y_values.append(x_data["cell_score"].values[0])

                if len(x_positions) == len(order):
                    ax.plot(x_positions, y_values, **default_line_params)

    if wilcoxon_test is not None:
        model_ids = stats_filt[kwargs["x"]].unique()
        order_levels = kwargs["order"]
        if not set(order_levels) == set(model_ids):
            raise ValueError(f"{kwargs['order']} must all in {kwargs['x']} column.")

        if "hue" in kwargs and "hue_order" in kwargs:
            hue_order_levels = kwargs["hue_order"]
            pairs = []
            for order_level in order_levels:
                pairs.append(
                    ((order_level, hue_order_levels[0]), (order_level, hue_order_levels[1]))
                )
            annotator = Annotator(
                ax,
                pairs,
                data=stats_filt,
                x=kwargs["x"],
                y="cell_score",
                order=order_levels,
                hue=kwargs["hue"],
                hue_order=hue_order_levels,
            )
        else:
            pairs = [tuple(order_levels)]
            annotator = Annotator(
                ax, pairs, data=stats_filt, x=kwargs["x"], y="cell_score", order=order_levels
            )

        print(pairs)

        annotator.configure(
            test=one_sided_wilcoxon(alternative=wilcoxon_test),
            show_test_name=False,
            text_format="star",
            verbose=1,
        )
        annotator.apply_and_annotate()

    if model_color_map is not None:
        for tick_label in ax.get_xticklabels():
            model_name = tick_label.get_text()
            color = model_color_map.get(model_name, "black")
            tick_label.set_color(color)

    ax.legend(title=None)

    lg_params = {} if lg_params is None else lg_params
    plt.legend(
        loc=lg_params.get("loc", "upper center"),
        bbox_to_anchor=lg_params.get("bbox_to_anchor", (0.5, -0.15)),
        ncol=lg_params.get("ncol", 4),
        frameon=lg_params.get("frameon", False),
        fancybox=lg_params.get("fancybox", False),
    )

    if new_model_names is not None:
        plt.xticks(range(len(new_model_names)), new_model_names)

    plt.xticks(rotation=xtick_rotation, ha=xtick_ha)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    plt.title(title_text)
    
    return ax


def vis_pval_barplot(
    stats_pval: pd.DataFrame,
    model_color_map: Optional[Dict[str, str]] = None,
    title_text: Optional[str] = None,
    xlabel_text: str = "-log10(p-value)",
    ylabel_text: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    new_model_names: Optional[List[str]] = None,
    lg_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Visualize the p-values (-log10 transformed) across given groups.

    Args:
        stats_pval: Statistics DataFrame with p-values.
        model_color_map: Dictionary mapping model names to colors.
        title_text: Plot title.
        xlabel_text: X-axis label (default: '-log10(p-value)').
        ylabel_text: Y-axis label.
        figsize: Figure size tuple (default: (8, 6)).
        new_model_names: List of new model names for y-axis.
        lg_params: Legend parameters dictionary.
        **kwargs: Additional arguments passed to seaborn barplot.

    Example:
        >>> vis_pval_barplot(stats_pval, y="model_id", hue="model_id")
    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=stats_pval, x="log_p_cut", **kwargs)

    threshold = -np.log10(0.05)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=0.7)

    if model_color_map is not None:
        for tick_label in ax.get_yticklabels():
            model_name = tick_label.get_text()
            color = model_color_map.get(model_name, "black")
            tick_label.set_color(color)

    if kwargs.get("y") == "model_id" and new_model_names is not None:
        plt.yticks(range(len(new_model_names)), new_model_names)

    lg_params = {} if lg_params is None else lg_params
    plt.legend(
        bbox_to_anchor=lg_params.get("bbox_to_anchor", (0.5, -0.15)),
        loc=lg_params.get("loc", "upper center"),
        ncol=lg_params.get("ncol", 1),
        frameon=lg_params.get("frameon", False),
        fancybox=lg_params.get("fancybox", False),
    )

    if kwargs.get("hue") == "log_p_cut":
        plt.legend().remove()

    plt.title(title_text)
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)


def vis_adjusted_delta_violin(
    stats_diff: pd.DataFrame,
    value_col: str = "cell_score_diff",
    group_col: str = "model_gene",
    wilcoxon_test_zero: Optional[str] = "greater",
    title_text: Optional[str] = None,
    xlabel_text: Optional[str] = None,
    ylabel_text: str = "Adjusted delta-sim score",
    new_model_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (5, 5),
    **kwargs: Any,
) -> None:
    """
    Visualize adjusted delta similarity distributions.

    Args:
        stats_diff: Statistics DataFrame with differences.
        value_col: Column name for values (default: 'cell_score_diff').
        group_col: Column name for groups (default: 'model_gene').
        wilcoxon_test_zero: Alternative hypothesis for Wilcoxon test against zero.
            None for no test.
        title_text: Plot title.
        xlabel_text: X-axis label.
        ylabel_text: Y-axis label (default: 'Adjusted delta-sim score').
        new_model_names: List of new model names for x-axis.
        figsize: Figure size tuple (default: (5, 5)).
        **kwargs: Additional arguments passed to seaborn violinplot.

    Example:
        >>> vis_adjusted_delta_violin(stats_diff, x="model_gene", y="cell_score_diff")
    """
    fig, ax = plt.subplots(figsize=figsize)
    violin_parts = sns.violinplot(
        data=stats_diff,
        x=group_col,
        y=value_col,
        hue=group_col,
        cut=2,
        inner="box",
        ax=ax,
        **kwargs,
    )

    for pc in violin_parts.collections:
        pc.set_edgecolor("white")
        pc.set_linewidth(0)

    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=1, zorder=0)

    # annotate significance
    model_genes = stats_diff[group_col].unique()
    if wilcoxon_test_zero is not None:
        y_data_max = stats_diff[value_col].max()
        y_data_min = stats_diff[value_col].min()
        y_range = y_data_max - y_data_min

        # Add extra space at the top (15-20% of range)
        y_limit_max = y_data_max + y_range * 0.2

        for i, gene in enumerate(model_genes):
            data = stats_diff[stats_diff[group_col] == gene][value_col].dropna()
            if len(data) > 0:
                # Wilcoxon test
                _, pval = wilcoxon(data, alternative=wilcoxon_test_zero)
                print(f"Wilcoxon test for {gene}: stat={_}, pval={pval}")
                # Determine significance stars
                if pval < 0.0001:
                    text = "****"
                elif pval < 0.001:
                    text = "***"
                elif pval < 0.01:
                    text = "**"
                elif pval < 0.05:
                    text = "*"
                else:
                    text = "ns"

                # Manual annotation
                y_max = data.max()
                y_range = stats_diff[value_col].max() - stats_diff[value_col].min()
                ax.text(
                    i,
                    y_max + y_range * 0.1,
                    text,
                    ha="center",
                    va="bottom",
                    color="gray",
                    fontsize=12,
                )

        ax.set_ylim(y_data_min - y_range * 0.2, y_limit_max)

    if new_model_names is not None:
        plt.xticks(range(len(new_model_names)), new_model_names)

    plt.xticks(rotation=30, ha="right")
    plt.xlabel(xlabel_text)
    plt.ylabel(ylabel_text)
    plt.title(title_text)


# =============================================================================
# Summary Utilities
# =============================================================================


def summary_isp_sim_stat_files(
    task_dir: Optional[str] = None,
    isp_type: str = "target_rank",
    model_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Summary of ISP conditions across models.

    Args:
        task_dir: Path to task directory.
        isp_type: Type of ISP analysis (default: 'target_rank').
        model_ids: List of model identifiers to include.

    Returns:
        Tuple containing:
            - isp_dict: Dictionary mapping conditions to model IDs.
            - condi_meta: DataFrame with condition metadata (perturb, config, condition).

    Raises:
        ValueError: If task_dir is not provided.

    Example:
        >>> isp_dict, condi_meta = summary_isp_sim_stat_files(task_dir, "target_rank")
    """
    if task_dir is None:
        raise ValueError("task_dir must be provided")

    task_stat_dir = Path(task_dir) / "stat"
    task_stat_models = [
        mid for mid in task_stat_dir.iterdir() if mid.name.startswith("GF")
    ]
    if model_ids is not None:
        task_stat_models = [
            mid for mid in task_stat_models if mid.name in model_ids
        ]

    task_stat_models = [mid / isp_type for mid in task_stat_models]
    task_stat_models = [mid for mid in task_stat_models if mid.exists()]

    isp_dict: Dict[str, List[str]] = defaultdict(list)
    for task_stat_model in task_stat_models:
        for file in task_stat_model.iterdir():
            isp_dict[file.stem].append(task_stat_model.parts[-2])

    isp_dict = dict(sorted(isp_dict.items()))

    conditions = list(isp_dict.keys())
    perturb = [condi.split("-", 5)[-1] for condi in conditions]
    config = [condi.split("-", 5)[-2] for condi in conditions]
    condi_meta = pd.DataFrame(
        {
            "perturb": perturb,
            "config": config,
            "condition": conditions,
        }
    )

    return isp_dict, condi_meta