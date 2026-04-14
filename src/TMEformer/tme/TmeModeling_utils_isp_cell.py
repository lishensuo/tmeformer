"""
TMEformer ISP Cell Utilities
============================

This module provides utilities for TME (Tumor Microenvironment) In-Silico Perturbation at cell level, including:
- TME composition perturbation
- TME rank perturbation
- Cell embedding extraction
- Multi-ISP context handling

Key Functions:
    - sample_with_replace_or_not: Sample cells with or without replacement
    - summarize_defaultdict: Calculate frequency of each item in a defaultdict
    - isp_tme_context: Core function for TME composition perturbation
    - multi_isp_tme_context: Perform multiple TME context perturbations
    - make_shifted_cell_dataset_and_embedding: Get perturbed cell dataset and embedding
    - perturb_one_tme_cells_composition: Main function for TME composition ISP
    - perturb_one_tme_cells_rank: Main function for TME rank ISP

Example:
    >>> # TME composition perturbation
    >>> shifted_dataset, shifted_emb = perturb_one_tme_cells_composition(
    ...     example_cell,
    ...     isp_cluster_id=1,
    ...     fixed_cluster_ids=None,
    ...     model=model,
    ...     cell_tme_dict=cell_tme_dict,
    ...     cell_cluster_dict=cell_cluster_dict,
    ...     isp_run_config=isp_run_config
    ... )
"""

import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk

from .. import TOKEN_DICTIONARY_FILE
from ..emb_extractor import get_embs
from .. import perturber_utils as pu
from . import TmeModeling_utils_isp_lst as tu_isp_lst
from . import TmeModeling_utils_isp_ds as tu_isp_ds
from . import TmeModeling_utils as tu


# =============================================================================
# TME Composition Perturbation Utilities
# =============================================================================


def sample_with_replace_or_not(
    cell_ids: List[int],
    size: int,
    seed: int,
) -> np.ndarray:
    """
    Sample cells with or without replacement.

    Args:
        cell_ids: Array of cell identifiers.
        size: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        Array of sampled cell IDs.

    Example:
        >>> sampled = sample_with_replace_or_not([1, 2, 3, 4, 5], size=10, seed=42)
    """
    np.random.seed(seed)
    if size > len(cell_ids):
        return np.random.choice(cell_ids, size=size, replace=True)
    else:
        return np.random.choice(cell_ids, size=size, replace=False)


def summarize_defaultdict(
    tme_map: Dict[int, List[int]],
    percent_format: bool = False,
) -> pd.DataFrame:
    """
    Calculate the frequency of each item (cluster) in a defaultdict.

    Args:
        tme_map: Dictionary mapping cluster IDs to lists of cell IDs.
        percent_format: If True, format frequencies as percentages.

    Returns:
        DataFrame with counts and frequencies sorted by index.

    Example:
        >>> df = summarize_defaultdict(tme_map, percent_format=True)
    """
    count_dict = {k: len(v) for k, v in tme_map.items()}
    df = pd.DataFrame.from_dict(count_dict, orient="index", columns=["Count"])
    df["Frequency"] = df["Count"] / df["Count"].sum()
    if percent_format:
        df["Frequency"] = (df["Frequency"] * 100).round(1).astype(str) + "%"
    return df.sort_index()


def isp_tme_context(
    tme_cells: List[int],
    cell_tme_dict: Dict[int, int],
    cell_cluster_dict: Dict[int, int],
    isp_cluster_id: int,
    goal_ratio: float,
    fixed_cluster_ids: Optional[List[int]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Core function for TME composition perturbation.

    Args:
        tme_cells: List of TME cell IDs.
        cell_tme_dict: Mapping from cell ID to main TME ID.
        cell_cluster_dict: Mapping from cell ID to cluster ID.
        isp_cluster_id: Target cluster ID for perturbation.
        goal_ratio: Target ratio for the perturbed cluster (>1 for OE, <1 for KD).
        fixed_cluster_ids: List of cluster IDs to keep unchanged.
        seed: Random seed for reproducibility.
        verbose: If True, print composition summaries.

    Returns:
        Tuple of (updated_cells, updated_types).

    Raises:
        ValueError: If isp_cluster_id not in cluster_ids.

    Example:
        >>> updated_cells, updated_types = isp_tme_context(
        ...     tme_cells=[1, 2, 3, 4, 5],
        ...     cell_tme_dict=cell_tme_dict,
        ...     cell_cluster_dict=cell_cluster_dict,
        ...     isp_cluster_id=1,
        ...     goal_ratio=1.5,
        ...     seed=42
        ... )
    """
    total_size = len(tme_cells)
    # {int: int} / {cell_id: main tme_id}
    id_cell2tme = {tme_cell: cell_tme_dict[tme_cell] for tme_cell in tme_cells}

    # {int: list} / {cluster_id: cell_id}
    id_cluster2cell = defaultdict(list)
    for tme_cell in tme_cells:
        id_cluster2cell[cell_cluster_dict[tme_cell]].append(tme_cell)

    cluster_ids = list(id_cluster2cell.keys())
    # {int: list} / {cluster_id: updated cell_id}
    updated_id_cluster2cell: Dict[int, List[int]] = {}

    # 1) Clusters not involved in ISP (keep unchanged)
    if fixed_cluster_ids is None:
        fixed_cluster_ids = []

    fixed_cell_count = 0
    for fixed_cluster_id in fixed_cluster_ids:
        fixed_cluster_cells = id_cluster2cell[fixed_cluster_id]
        fixed_cell_count += len(fixed_cluster_cells)
        updated_id_cluster2cell[fixed_cluster_id] = fixed_cluster_cells

    # 2) Clusters directly involved in ISP
    # Check if isp_cluster_id in cluster_ids
    if isp_cluster_id not in cluster_ids:
        print(f"### Targeted isp_cluster_id: {isp_cluster_id}")
        print(f"### Existed cluster_ids: {cluster_ids}")
        raise ValueError(
            "isp_cluster_id not in cluster_ids "
            "(The cell has no such TME cluster cells)"
        )

    isp_cluster_cells = id_cluster2cell[isp_cluster_id]
    # goal_ratio could be >1 (OE/Expand) or <1 (KD/Delete)
    goal_size = int(len(isp_cluster_cells) * goal_ratio)
    if goal_size > total_size - fixed_cell_count:
        goal_size = total_size - fixed_cell_count

    updated_isp_cluster_cells = sample_with_replace_or_not(
        isp_cluster_cells, goal_size, seed
    )
    updated_id_cluster2cell[isp_cluster_id] = updated_isp_cluster_cells

    # 3) Clusters indirectly involved in ISP (keep total TME cells unchanged)
    remain_size = total_size - goal_size - fixed_cell_count

    aff_cluster_ids = list(
        set(cluster_ids) - set([isp_cluster_id] + fixed_cluster_ids)
    )
    aff_cells = [
        cell_id for k in aff_cluster_ids for cell_id in id_cluster2cell[k]
    ]

    updated_aff_cells = sample_with_replace_or_not(aff_cells, remain_size, seed)
    updated_aff_clusters: Dict[int, List[int]] = defaultdict(list)
    for tme_cell in updated_aff_cells:
        updated_aff_clusters[cell_cluster_dict[tme_cell]].append(tme_cell)

    updated_id_cluster2cell.update(updated_aff_clusters)

    # TME composition summary
    if verbose:
        print("### Before Adjustment:")
        print(summarize_defaultdict(id_cluster2cell))
        print("### After Adjustment:")
        print(summarize_defaultdict(updated_id_cluster2cell))

    # List of cell_ids
    updated_cells = [
        cell for cells in list(updated_id_cluster2cell.values()) for cell in cells
    ]
    # List of tme_ids
    updated_types = [id_cell2tme[cell] for cell in updated_cells]

    return updated_cells, updated_types


def multi_isp_tme_context(
    tme_cells: List[int],
    cell_tme_dict: Dict[int, int],
    cell_cluster_dict: Dict[int, int],
    isp_cluster_id: int,
    endpoint: float = 1,
    window: float = 0,
    fixed_cluster_ids: Optional[List[int]] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Perform multiple TME context perturbations with different goal ratios.

    Args:
        tme_cells: List of TME cell IDs.
        cell_tme_dict: Mapping from cell ID to main TME ID.
        cell_cluster_dict: Mapping from cell ID to cluster ID.
        isp_cluster_id: Target cluster ID for perturbation.
        endpoint: Target endpoint ratio.
        window: Step size for generating ratios (0 for endpoint-only).
        fixed_cluster_ids: List of cluster IDs to keep unchanged.
        seed: Random seed for reproducibility.
        verbose: If True, print composition summaries.

    Returns:
        Dictionary mapping offsets to perturbed TME data.

    Raises:
        ValueError: If endpoint and window are incompatible.

    Example:
        >>> isp_tme_dicts = multi_isp_tme_context(
        ...     tme_cells=[1, 2, 3, 4, 5],
        ...     cell_tme_dict=cell_tme_dict,
        ...     cell_cluster_dict=cell_cluster_dict,
        ...     isp_cluster_id=1,
        ...     endpoint=2.0,
        ...     window=0.5
        ... )
    """
    if window != 0:
        # Area-based ISP
        if endpoint < 1 and window > 0:
            raise ValueError("endpoint must be >= 1 if window > 0")
        if endpoint > 1 and window < 0:
            raise ValueError("endpoint must be <= 1 if window < 0")
        goal_ratios = np.unique(
            np.append(np.round(np.arange(1, endpoint, window), 2), endpoint)
        )
    else:
        # Endpoint-based ISP
        goal_ratios = np.unique(np.array([1, endpoint], dtype=np.float64))

    # For each goal_ratio, adjust the corresponding TME composition
    new_cells_list, new_types_list = [], []
    for goal_ratio in goal_ratios:
        if goal_ratio == 1:
            # No ISP: keep unchanged
            new_cells = tme_cells
            new_types = [cell_tme_dict[tme_cell] for tme_cell in tme_cells]
        else:
            # ISP: adjust TME composition
            new_cells, new_types = isp_tme_context(
                tme_cells,
                cell_tme_dict,
                cell_cluster_dict,
                isp_cluster_id,
                goal_ratio,
                fixed_cluster_ids,
                seed,
                verbose=verbose,
            )
        new_cells_list.append(new_cells)
        new_types_list.append(new_types)

    new_tme_cells = np.array(new_cells_list)
    new_tme_types = np.array(new_types_list)

    # Keep 2-dimensional array
    if len(new_tme_cells.shape) == 1:
        new_tme_cells = new_tme_cells.reshape(-1, 1)
    if len(new_tme_types.shape) == 1:
        new_tme_types = new_tme_types.reshape(-1, 1)

    # {int: {str: list}}
    isp_tme_dicts: Dict[float, Dict[str, np.ndarray]] = defaultdict(dict)
    for goal_ratio, new_tme_cell, new_tme_type in zip(
        goal_ratios, new_tme_cells, new_tme_types
    ):
        # offset > 0: OE(Expand), offset < 0: KD(Delete)
        offset = goal_ratio - 1
        # Note: when gene rank ISP, offset = init_rank - new_rank
        # Smaller rank means larger expression, so offset > 0 also means OE
        isp_tme_dicts[offset]["tme_cells"] = new_tme_cell
        isp_tme_dicts[offset]["tme_types"] = new_tme_type

    return isp_tme_dicts


def make_shifted_cell_dataset_and_embedding(
    example_cell: Dataset,
    isp_tme_dicts: Dict[float, Dict[str, np.ndarray]],
    model: Any,
    embed_layer: int = -1,
    batch_size: int = 10,
    do_embed: bool = True,
) -> Tuple[Dataset, Optional[torch.Tensor]]:
    """
    Get the perturbed cell dataset and embedding for a given cell.

    Args:
        example_cell: Example cell data.
        isp_tme_dicts: Dictionary of ISP perturbation results.
        model: Model for embedding extraction.
        embed_layer: Layer index for embedding extraction.
        batch_size: Batch size for embedding computation.
        do_embed: If True, compute embeddings.

    Returns:
        Tuple of (shifted_cell_dataset, shifted_cell_emb).

    Example:
        >>> dataset, emb = make_shifted_cell_dataset_and_embedding(
        ...     example_cell, isp_tme_dicts, model, embed_layer=-1
        ... )
    """
    offsets = list(isp_tme_dicts.keys())
    shifted_cell_dataset = Dataset.from_dict(
        {
            "input_ids": example_cell["input_ids"] * len(offsets),
            "length": example_cell["length"] * len(offsets),
            "offset": offsets,
            "cell_id": example_cell["cell_id"] * len(offsets),
            "sample_id": example_cell["sample_id"] * len(offsets),
            "tme_cells": np.array(
                [isp_tme_dicts[offset]["tme_cells"] for offset in offsets]
            ),
            "tme_types": np.array(
                [isp_tme_dicts[offset]["tme_types"] for offset in offsets]
            ),
        }
    )

    shifted_cell_emb = None
    if do_embed:
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            gene_token_dict = pickle.load(f)
        token_gene_dict = {v: k for k, v in gene_token_dict.items()}

        pad_token_id = gene_token_dict.get("<pad>")
        layer_to_quant = pu.quant_layers(model) + embed_layer

        shifted_cell_emb = get_embs(
            model,
            shifted_cell_dataset,
            "cell",
            layer_to_quant,
            pad_token_id,
            batch_size,
            token_gene_dict=token_gene_dict,
            summary_stat=None,
            silent=True,
        ).to("cpu")

    return shifted_cell_dataset, shifted_cell_emb


# =============================================================================
# TME ISP Main Functions
# =============================================================================


def perturb_one_tme_cells_composition(
    example_cell: Dataset,
    isp_cluster_id: int,
    fixed_cluster_ids: Optional[List[int]],
    model: Any,
    cell_tme_dict: Dict[int, int],
    cell_cluster_dict: Dict[int, int],
    isp_run_config: Dict[str, Any],
    embed_layer: int = -1,
    batch_size: int = 10,
) -> Tuple[Dataset, torch.Tensor]:
    """
    Main function to perform TME composition ISP on a given cell.

    Args:
        example_cell: Example cell data.
        isp_cluster_id: Target cluster ID for perturbation.
        fixed_cluster_ids: List of cluster IDs to keep unchanged.
        model: Model for embedding extraction.
        cell_tme_dict: Mapping {cell_id: main tme_id}.
        cell_cluster_dict: Mapping {cell_id: finer cluster_id}.
        isp_run_config: Configuration dict with 'endpoints' and 'window'.
        embed_layer: Layer index for embedding extraction.
        batch_size: Batch size for embedding computation.

    Returns:
        Tuple of (shifted_cell_dataset, shifted_cell_emb).

    Note:
        Main times cell_tme_dict = cell_cluster_dict

    Example:
        >>> shifted_dataset, shifted_emb = perturb_one_tme_cells_composition(
        ...     example_cell,
        ...     isp_cluster_id=1,
        ...     fixed_cluster_ids=None,
        ...     model=model,
        ...     cell_tme_dict=cell_tme_dict,
        ...     cell_cluster_dict=cell_cluster_dict,
        ...     isp_run_config=isp_run_config
        ... )
    """
    endpoints = isp_run_config.get("endpoints", 0)
    window = isp_run_config.get("window", 0)

    tme_cells = example_cell[0]["tme_cells"]

    isp_tme_dicts = multi_isp_tme_context(
        tme_cells,
        cell_tme_dict,
        cell_cluster_dict,
        isp_cluster_id,
        endpoint=endpoints,
        window=window,
        fixed_cluster_ids=fixed_cluster_ids,
    )

    shifted_cell_dataset, shifted_cell_emb = make_shifted_cell_dataset_and_embedding(
        example_cell,
        isp_tme_dicts,
        model,
        embed_layer,
        batch_size,
    )

    return shifted_cell_dataset, shifted_cell_emb


def perturb_one_tme_cells_rank(
    example_cell: Dataset,
    isp_cluster_id: int,
    gene_tokens: List[int],
    model: Any,
    cell_cluster_dict: Dict[int, int],
    isp_run_config: Optional[Dict[str, Any]] = None,
    resample: bool = True,
    kd_only_resample: bool = False,
    emb_type: str = "cell",
    cell_embed_layer: int = -1,
    work_dir: Optional[str] = None,
    proj: str = "xenium",
) -> Tuple[Dataset, Optional[torch.Tensor]]:
    """
    Main function to perform TME rank ISP on a given cell.

    Args:
        example_cell: Example cell data.
        isp_cluster_id: Target cluster ID for perturbation.
        gene_tokens: Target gene tokens for perturbation.
        model: Model for embedding extraction.
        cell_cluster_dict: Mapping from cell ID to cluster ID.
        isp_run_config: Configuration dict for ISP.
        resample: If True, resample non-expressing cells.
        kd_only_resample: If True, use special resampling for KD.
        emb_type: Type of embedding to return ('cell' or 'tme').
        cell_embed_layer: Layer index for cell embedding extraction.
        work_dir: Working directory path.
        proj: Project name.

    Returns:
        Tuple of (example_cells, embeddings).

    Raises:
        ValueError: If OE/KD isp_type has invalid offset.

    Example:
        >>> example_cells, embeddings = perturb_one_tme_cells_rank(
        ...     example_cell,
        ...     isp_cluster_id=1,
        ...     gene_tokens=[1, 2, 3],
        ...     model=model,
        ...     cell_cluster_dict=cell_cluster_dict,
        ...     resample=True,
        ...     emb_type="cell"
        ... )
    """
    example_cells = concatenate_datasets([example_cell] * 3)  # Copy 3 times
    example_cells = example_cells.add_column("offset", [0, -1, 1])  # [Vanilla, KD, OE]

    tme_cells = np.array(example_cell[0]["tme_cells"])
    tme_clusters = np.array(
        [cell_cluster_dict[tme_cell] for tme_cell in tme_cells]
    )

    gf_cell_embeddings = np.load(
        work_dir + model.config.tme_config["gf_tme_emb_path"] + "cell_embed.npy",
        mmap_mode="r",
    )
    dataset_all = load_from_disk(work_dir + f"data/{proj}/datasets/{proj}.dataset")
    PR_XE_MODELS_DICT = tu.generate_pr_models_dict("checkpoint_xe", work_dir)
    model_cl = pu.load_model(
        "Pretrained_TME",
        model_directory=PR_XE_MODELS_DICT["GF_CL"][0],
        mode="eval",
        device=model.device,
    )

    tme_embs_init = gf_cell_embeddings[tme_cells - 1]
    cells2tme_embs_init = dict(zip(tme_cells, tme_embs_init))

    # TME cells subset that belong to the target cluster
    cluster_tme_cells = [
        tme_cells[i] for i, cluster in enumerate(tme_clusters) if cluster == isp_cluster_id
    ]
    dataset_clu_cells = dataset_all.select(np.array(cluster_tme_cells) - 1)

    # Further filter TME cells subset that express the target gene
    dataset_clu_cells_filt = tu_isp_ds.filter_token_dataset(
        dataset_clu_cells, tokens=gene_tokens, existed=True
    )
    if len(dataset_clu_cells_filt) == 0:
        return example_cells, None

    # Cell IDs of TME cells subset that express the target gene
    shifted_cells = dataset_clu_cells_filt["cell_id"]
    # Cell IDs of TME cells subset that don't express the target gene
    remained_cells = list(set(dataset_clu_cells["cell_id"]) - set(shifted_cells))

    shifted_cell_embs_dict: Dict[str, Dict[int, np.ndarray]] = {"OE": {}, "KD": {}}
    for i in range(len(shifted_cells)):
        tme_cell = dataset_clu_cells_filt.select([i])

        # Endpoint ISP for each TME cell that expresses the target gene
        if isp_run_config is None:
            isp_run_config = {
                "window": 0,
                "endpoints": 1,
                "ko_method": None,
                "ki_method": None,
            }
        shifted_cell_dataset, shifted_cell_emb = tu_isp_lst.perturb_one_cell(
            tme_cell,
            gene_tokens,
            model_cl,
            isp_run_config,
            batch_size=3,
            embed_layer=0,  # Layer 0 for TME embedding
        )
        # 3 cells at most for shifted_cell_dataset due to endpoint ISP
        # 2 cells in extreme condition (target gene already at endpoint)

        for isp_type in ["OE", "KD"]:
            # >=0: OE; <=0: KD; =0: Non
            if isp_type == "OE":
                offset_arg = np.argmax(shifted_cell_dataset["offset"])
                offset_val = shifted_cell_dataset["offset"][offset_arg]
                if offset_val < 0:
                    raise ValueError("OE isp_type must have positive offset")
            elif isp_type == "KD":
                offset_arg = np.argmin(shifted_cell_dataset["offset"])
                offset_val = shifted_cell_dataset["offset"][offset_arg]
                if offset_val > 0:
                    raise ValueError("KD isp_type must have negative offset")

            # {str('OE'/'KD'): {int(cell_id): np.array(embedding)}}
            shifted_cell_embs_dict[isp_type][tme_cell["cell_id"][0]] = shifted_cell_emb[
                offset_arg
            ]

    # For TME cells in target cluster that don't express target gene,
    # use embedding of TME cells with expression after ISP instead
    if resample and len(remained_cells) > 0:
        np.random.seed(42)
        resampled_cells = np.random.choice(
            shifted_cells, size=len(remained_cells), replace=True
        )
        for isp_type in ["OE", "KD"]:
            for remained_cell, resampled_cell in zip(
                remained_cells, resampled_cells
            ):
                shifted_cell_embs_dict[isp_type][remained_cell] = (
                    shifted_cell_embs_dict[isp_type][resampled_cell]
                )

    if kd_only_resample and len(shifted_cells) > 0:
        # Directly replace gene-expressing TME cells with non-expressing cells
        shifted_cell_embs_dict["KD"].clear()
        np.random.seed(42)
        resampled_cells = np.random.choice(
            remained_cells, size=len(shifted_cells), replace=True
        )
        for shifted_cell, resampled_cell in zip(shifted_cells, resampled_cells):
            shifted_cell_embs_dict["KD"][shifted_cell] = cells2tme_embs_init[
                resampled_cell
            ]

    # Update TME cell embeddings after ISP
    tme_embs_kd = tme_embs_init.copy()
    tme_embs_oe = tme_embs_init.copy()
    for tme_cell, shifted_embedding in shifted_cell_embs_dict["KD"].items():
        tme_cell_arg = tme_cells.tolist().index(tme_cell)
        tme_embs_kd[tme_cell_arg] = shifted_embedding
    for tme_cell, shifted_embedding in shifted_cell_embs_dict["OE"].items():
        tme_cell_arg = tme_cells.tolist().index(tme_cell)
        tme_embs_oe[tme_cell_arg] = shifted_embedding

    # Keep order same with "offset": [0, -1, 1]
    tme_embs_new = np.stack([tme_embs_init, tme_embs_kd, tme_embs_oe], axis=0)
    tme_embs_new = torch.tensor(tme_embs_new)

    if emb_type == "tme":
        # Use for GEP task prediction
        return example_cells, tme_embs_new

    # Update target cell embedding after its TME cells are affected
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    token_gene_dict = {v: k for k, v in gene_token_dict.items()}
    pad_token_id = gene_token_dict.get("<pad>")
    layer_to_quant = pu.quant_layers(model) + cell_embed_layer

    cell_emb = get_embs(
        model,
        example_cells,
        "cell",
        layer_to_quant,
        pad_token_id,
        forward_batch_size=3,
        token_gene_dict=token_gene_dict,
        summary_stat=None,
        silent=True,
        tme_cell_embs=tme_embs_new,
    ).to("cpu")

    if emb_type == "cell":
        # Use for similarity task prediction
        return example_cells, cell_emb

    return example_cells, cell_emb