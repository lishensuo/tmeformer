"""
TMEformer ISP Dataset Utilities
================================

This module provides utilities for processing TME datasets, including:
- In-silico perturbation dataset filtering
- ISP score calculation
- Dataset sampling and interval filtering
- Model and dataset loading

Key Functions:
    - group_ft_dataset: Process HuggingFace dataset and add group annotations
    - embed_ft_dataset: Extract and average cell embeddings per state group
    - interval_dataset: Filter cells by gene expression percentile interval
    - filter_token_dataset: Filter dataset by gene token presence
    - filter_tme_composition_dataset: Filter dataset by TME composition
    - filter_tme_rank_dataset: Filter dataset by TME rank
    - calc_isp_score: Calculate ISP score from perturbation results
    - load_model_and_dataset: Load pre-trained model and dataset

Example:
    >>> dataset = filter_token_dataset(dataset, state_dict={"Tumor": ["C-001-T"]}, tokens=[1, 2, 3])
    >>> score = calc_isp_score(cell_id_dict, score_type="OE", score_method="area")
"""

import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_caching, load_from_disk

from .. import TOKEN_DICTIONARY_FILE, EmbExtractor
from .. import perturber_utils as pu
from . import TmeModeling_utils as tu


# =============================================================================
# Dataset Grouping and Embedding
# =============================================================================


def group_ft_dataset(
    sample_meta_groups: Dict[str, List],
    sample_dict_file: str,
    ds_raw_file: Union[str, Dataset],
    ds_output_file: Optional[str] = None,
    sample_max_cells: int = 1000,
) -> Dataset:
    """
    Process HuggingFace dataset: extract cells belonging to groups and add annotations.

    Args:
        sample_meta_groups: Dictionary of sample groupings.
            Example:
            {
                "Tumor": [("C-038-T",), ("C-039-T",)],
                "Normal": [("C-038-N",), ("C-039-N",)],
                "PID": ["C-038", "C-039"]
            }
        sample_dict_file: Path to {sample_name: sample_id} mapping (pickle file).
            e.g., "data/xenium/processed/xenium_sample_id_dict.pkl"
        ds_raw_file: Path to the original HuggingFace dataset or Dataset object.
            e.g., "data/xenium/datasets/xenium_epi_TME_v4.dataset"
            Must have sample_id field.
        ds_output_file: Optional output path to save the processed dataset.
        sample_max_cells: Maximum number of cells to sample per sample.

    Returns:
        Group subset dataset with extra annotations ("group", "sample", "patient").

    Raises:
        ValueError: If some sample names are missing in sample_dict_file.

    Example:
        >>> sample_meta_groups = {
        ...     "Tumor": [("C-038-T",), ("C-039-T",)],
        ...     "Normal": [("C-038-N",), ("C-039-N",)],
        ...     "PID": ["C-038", "C-039"]
        ... }
        >>> dataset = group_ft_dataset(
        ...     sample_meta_groups,
        ...     sample_dict_file="data/xenium/processed/xenium_sample_id_dict.pkl",
        ...     ds_raw_file="data/xenium/datasets/xenium_epi_TME_v4.dataset"
        ... )
    """
    if ds_output_file is not None and os.path.exists(ds_output_file):
        print(f"Dataset {ds_output_file} already exists")
        return load_from_disk(ds_output_file)

    with open(sample_dict_file, "rb") as f:
        spname2id = pickle.load(f)

    sample_to_pid = {}  # {sample_name: patient_name}
    for group, samples in sample_meta_groups.items():
        if group == "PID":
            continue
        for i, sample_group in enumerate(samples):
            for sample in sample_group:
                sample_to_pid[sample] = sample_meta_groups["PID"][i]

    def flatten(lst):
        return [x for xx in lst for x in xx]

    # e.g. {'Tumor': ['C-038-T', 'C-039-T'], 'Normal': ['C-038-N', 'C-039-N']}
    sample_groups = {
        k: flatten(v) for k, v in sample_meta_groups.items() if k != "PID"
    }

    pool_id2group = {
        spname2id[sample]: group
        for group, samples in sample_groups.items()
        for sample in samples
        if sample in spname2id
    }

    pool_id2sample = {
        spname2id[sample]: sample
        for samples in sample_groups.values()
        for sample in samples
        if sample in spname2id
    }

    pool_id2patient = {
        spname2id[sample]: sample_to_pid[sample]
        for sample in sample_to_pid
        if sample in spname2id
    }

    if not set(pool_id2group.keys()).issubset(set(spname2id.values())):
        raise ValueError("Some sample names are missing in sample_dict_file!")

    if isinstance(ds_raw_file, str):
        dataset = load_from_disk(ds_raw_file)
    else:
        dataset = ds_raw_file

    # {sample_id: [row_index]}
    sample_to_indices = (
        pd.DataFrame({"sample_id": dataset["sample_id"]})
        .reset_index()
        .groupby("sample_id")["index"]
        .apply(list)
        .to_dict()
    )

    selected_indices = []
    for sample_id, indices in sample_to_indices.items():
        if sample_id in pool_id2group:
            if len(indices) >= sample_max_cells:
                random.seed(42)
                sampled = random.sample(indices, sample_max_cells)
                selected_indices.extend(sampled)
            else:
                selected_indices.extend(indices)

    # Dataset subset within group samples
    dataset = dataset.select(selected_indices)

    # Add group column
    if "group" not in dataset.column_names:
        dataset = dataset.map(
            lambda x: {"group": pool_id2group[x["sample_id"]]}, num_proc=10
        )
    # Add patient column
    dataset = dataset.map(
        lambda x: {"patient": pool_id2patient.get(x["sample_id"])}, num_proc=10
    )
    # Add sample column
    dataset = dataset.map(
        lambda x: {"sample": pool_id2sample.get(x["sample_id"])}, num_proc=10
    )

    if ds_output_file is not None:
        dataset.save_to_disk(ds_output_file)
        print(f"Dataset saved to {ds_output_file}")

    return dataset


def embed_ft_dataset(
    task_embed_dir: str,
    task_dataset_dir: str,
    sample_meta_groups: Dict[str, List],
    model_id: str = "GF_PR",
    model_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Extract and average cell embeddings per state group.

    Args:
        task_embed_dir: Directory to save the output embeddings.
        task_dataset_dir: Directory to the task HuggingFace dataset.
        sample_meta_groups: Sample groupings.
            Example: {"Tumor": [("C-038-T",)], "PID": [("C-038",)]}
        model_id: Model identifier, must be key in model_dict.
        model_config: Configuration dict with keys:
            - model_dict: Dictionary mapping model_id to (path, version, type)
            - embed_layer: Layer index for cell embeddings
            - device: GPU device ID
            - batch_size: Batch size for embedding extraction

    Raises:
        FileNotFoundError: If required dataset files not found.
        ValueError: If embed_out_dir does not exist.
    """
    model_dict = model_config["model_dict"]
    model_id_item = model_dict[model_id]

    check_file_1 = f"{task_dataset_dir}/{model_id_item[1]}/isp_to.dataset"
    check_file_2 = f"{task_dataset_dir}/{model_id_item[1]}/isp.dataset"
    if os.path.exists(check_file_1):
        dataset_file = check_file_1
    elif os.path.exists(check_file_2):
        dataset_file = check_file_2
    else:
        raise FileNotFoundError(f"{check_file_1} or {check_file_2} not found")

    print(f"## {dataset_file} is used to calc goal embedding")

    embed_out_dir = f"{task_embed_dir}/{model_id}"
    embed_out_file = (
        f"{embed_out_dir}/L{abs(model_config['embed_layer'])}_states_mean.pickle"
    )

    if not os.path.exists(embed_out_dir):
        raise ValueError(f"{embed_out_dir} not exists")

    if os.path.exists(embed_out_file):
        print(f"Embedding file {embed_out_file} already exists")
        return None

    # Initialize embedding extractor
    embex = EmbExtractor(
        model_type="Pretrained_TME",
        emb_mode="cell",
        emb_layer=model_config["embed_layer"],
        forward_batch_size=model_config["batch_size"],
        nproc=16,
        max_ncells=None,
        token_dictionary_file=str(TOKEN_DICTIONARY_FILE),
        device=f"cuda:{model_config['device']}",
    )

    # Run embedding extraction
    embs = embex.extract_embs(
        model_directory=model_id_item[0],
        input_data_file=dataset_file,
        output_directory=embed_out_dir,
        output_prefix=f"L{abs(model_config['embed_layer'])}_states_raw",
    )

    # Load sample info from dataset
    dataset = load_from_disk(dataset_file)
    dataset_sample = dataset.select_columns(["sample"]).to_pandas()

    states_mean_dict = defaultdict(list)
    # Mean embedding for each sample, like {group_name: [1-D array]}
    for group, samples in sample_meta_groups.items():
        if group != "PID":
            for sample in samples:
                state_mean = (
                    embs[dataset_sample["sample"].isin(sample)]
                    .mean(axis=0)
                    .values
                )
                states_mean_dict[group].append(state_mean)

    with open(embed_out_file, "wb") as f:
        pickle.dump(states_mean_dict, f)


# =============================================================================
# Dataset Filtering Utilities
# =============================================================================


def interval_dataset(
    dataset: Dataset,
    tokens: List[int],
    interval: Optional[List[float]] = None,
) -> Dataset:
    """
    Filter cells with gene token(s) at a specific percentile interval of expression level.

    Args:
        dataset: HuggingFace Dataset object with an "input_ids" field.
        tokens: List of gene token IDs to check. Must exist in every input_ids.
        interval: List or tuple [lower_bound, upper_bound] where values are in [0, 1].
            - [0, 0.5]: Keep higher 50% expression level
            - [0.5, 1]: Keep lower 50% expression level

    Returns:
        Filtered HuggingFace Dataset object.

    Raises:
        ValueError: If interval format is invalid.

    Example:
        >>> dataset = interval_dataset(dataset, tokens=[1, 2, 3], interval=[0, 0.5])
    """
    if interval is None or interval == [0, 1]:
        return dataset

    lower_bound, upper_bound = interval

    # For KD, to keep high-expressed cells
    # e.g. [0, 0.5], token pct_rank <= 0.5 or tokens max(pct_rank) <= 0.5
    if lower_bound == 0:
        dataset = dataset.filter(
            function=lambda batch: [
                max(x.index(one_token) for one_token in tokens) / len(x) <= upper_bound
                for x in batch["input_ids"]
            ],
            batched=True,
            batch_size=1000,
        )

    # For OE, to keep low-expressed cells
    # e.g. [0.5, 1], token pct_rank >= 0.5 or tokens min(pct_rank) >= 0.5
    elif upper_bound == 1:
        dataset = dataset.filter(
            function=lambda batch: [
                min(x.index(one_token) for one_token in tokens) / len(x)
                >= lower_bound
                for x in batch["input_ids"]
            ],
            batched=True,
            batch_size=1000,
        )
    else:
        raise ValueError(
            "Interval should be either [0, 1] or [0, x] or [x, 1]"
        )

    return dataset


def tme_interval_dataset(
    dataset: Dataset,
    interval: Optional[List[float]] = None,
) -> Dataset:
    """
    Filter cell datasets according to the 'ratio' column.

    Args:
        dataset: HuggingFace Dataset with a 'ratio' column.
        interval: Ratio interval [lower_bound, upper_bound].

    Returns:
        Filtered dataset.

    Raises:
        ValueError: If interval format is invalid.
    """
    if interval is None or interval == [0, 1]:
        return dataset

    lower_bound, upper_bound = interval

    # e.g. [0, 0.5]
    if lower_bound == 0:
        dataset = dataset.filter(
            function=lambda batch: [upper_bound > x for x in batch["ratio"]],
            batched=True,
            batch_size=1000,
        )

    # e.g. [0.5, 1]
    elif upper_bound == 1:
        dataset = dataset.filter(
            function=lambda batch: [lower_bound < x for x in batch["ratio"]],
            batched=True,
            batch_size=1000,
        )
    else:
        raise ValueError(
            "Interval should be either [0, 1] or [0, x] or [x, 1]"
        )

    return dataset


def sample_dataset(
    dataset: Dataset,
    ratio: Optional[Union[int, float]] = None,
) -> Dataset:
    """
    Randomly sample a subset of the dataset based on a given ratio or fixed number.

    Args:
        dataset: HuggingFace Dataset object.
        ratio:
            - If float (0 < ratio < 1): sample a proportion of the dataset.
            - If int (ratio > 1): sample a fixed number of examples.
            - If None: return the original dataset.

    Returns:
        A sampled subset of the dataset.

    Raises:
        ValueError: If ratio format is invalid.
    """
    if ratio is None:
        return dataset

    total_samples = len(dataset)

    if isinstance(ratio, (int, float)):
        if isinstance(ratio, int) and ratio > 1:
            # Sample a fixed number of examples
            num_samples = min(ratio, total_samples)
            return dataset.shuffle(seed=42).select(range(num_samples))

        elif isinstance(ratio, float) and 0 < ratio < 1:
            # Sample a proportion of the dataset
            return dataset.train_test_split(test_size=ratio, seed=42)["test"]

    raise ValueError(
        "ratio must be either a positive integer or a float between 0 and 1"
    )


def filter_token_dataset(
    dataset: Dataset,
    state_dict: Optional[Dict[str, List[str]]] = None,
    tokens: Optional[List[int]] = None,
    existed: bool = True,
    interval: Optional[List[float]] = None,
    ratio: Optional[Union[int, float]] = None,
) -> Dataset:
    """
    Filter a dataset to keep cells that express or do not express specific gene(s).

    Args:
        dataset: A HuggingFace Dataset object with fields "group", "sample", "input_ids".
        state_dict: Dictionary mapping {group_name: sample_name} to filter.
            e.g., {"Tumor": ["C-001-T", "C-002-T"]}.
        tokens: List of token IDs representing target genes.
        existed:
            - True: Keep cells that express all tokens.
            - False: Keep cells that do not express any of the tokens.
        interval: e.g. [0, 0.5] for high-expressed cells or [0.5, 1] for low-expressed cells.
            Only used when existed=True.
        ratio:
            - If float (0 < ratio < 1): randomly sample a proportion of cells.
            - If int (ratio > 1): randomly sample a fixed number of cells.

    Returns:
        A filtered HuggingFace Dataset object.

    Raises:
        ValueError: If state_dict contains more than one key or no matching cells found.
    """
    # Validate and extract group filter
    if state_dict is not None:
        if not state_dict or len(state_dict) != 1:
            raise ValueError("state_dict must contain exactly one key-value pair.")

        group, samples = list(state_dict.items())[0]

        # Step 1: Filter by group_name and sample_name
        dataset = dataset.filter(
            function=lambda batch: [
                (g == group and s in samples)
                for g, s in zip(batch["group"], batch["sample"])
            ],
            batched=True,
            batch_size=1000,
        )
        if len(dataset) == 0:
            raise ValueError(
                "No matching cells found for the specified group and sample IDs."
            )

    # Step 2: Filter based on presence or absence of tokens
    disable_caching()

    if tokens is not None:
        if existed:
            # Keep only cells where all target tokens are present
            token_dataset = dataset.filter(
                function=lambda batch: [
                    set(tokens).issubset(x) for x in batch["input_ids"]
                ],
                batched=True,
                batch_size=1000,
                num_proc=4 if len(dataset) > 10000 else 1,
            )
            # Optional: Further filter by expression rank interval
            if interval is not None and interval not in ([0, 1], (0, 1)):
                token_dataset = interval_dataset(token_dataset, tokens, interval)
        else:
            # Keep only cells where none of the tokens are present
            token_dataset = dataset.filter(
                function=lambda batch: [
                    set(tokens).isdisjoint(x) for x in batch["input_ids"]
                ],
                batched=True,
                batch_size=1000,
                num_proc=4 if len(dataset) > 10000 else 1,
            )

    # Step 3: Optionally sample a subset of the filtered dataset
    if ratio is not None:
        token_dataset = sample_dataset(token_dataset, ratio)

    return token_dataset


def batch_exceed_total_cells_filter(
    batch: Dict[str, Any],
    isp_cluster: int,
    endpoints: float,
) -> List[bool]:
    """
    Check whether expansion of one cluster will exceed the maximum of TME cells.

    Args:
        batch: Batch data with 'cluster_types' field.
        isp_cluster: Target cluster ID.
        endpoints: Expansion endpoint ratio.

    Returns:
        List of boolean values indicating whether each cell passes the filter.
    """
    results = []
    for tme_list in batch["cluster_types"]:
        total_cells = len(tme_list)
        counter = Counter(tme_list)
        if counter[isp_cluster] * endpoints >= total_cells:
            results.append(False)
        else:
            results.append(True)
    return results


def filter_tme_composition_dataset(
    dataset: Dataset,
    state_dict: Optional[Dict[str, List[str]]] = None,
    isp_cluster_id: Optional[int] = None,
    fixed_cluster_ids: Optional[List[int]] = None,
    existed: bool = True,
    interval: Optional[List[float]] = None,
    ratio: Optional[Union[int, float]] = None,
    cluster_endpoint: float = 1,
) -> Dataset:
    """
    Filter cell dataset whose tme_cells can be implemented with TME_Composition ISP.

    Args:
        dataset: HuggingFace Dataset object.
        state_dict: Dictionary for group/sample filtering.
        isp_cluster_id: Target cluster ID for perturbation.
        fixed_cluster_ids: List of cluster IDs to keep unchanged.
        existed: Whether to filter for presence or absence of cluster.
        interval: Compositional ratio interval.
        ratio: Sampling ratio.
        cluster_endpoint: Expansion endpoint for OE perturbation.

    Returns:
        Filtered dataset.

    Raises:
        ValueError: If state_dict contains more than one key or no matching cells found.
    """
    # Step 1: Validate and extract group filter
    if state_dict is not None:
        if not state_dict or len(state_dict) != 1:
            raise ValueError("state_dict must contain exactly one key-value pair.")

        group, samples = list(state_dict.items())[0]

        dataset = dataset.filter(
            function=lambda batch: [
                (g == group and s in samples)
                for g, s in zip(batch["group"], batch["sample"])
            ],
            batched=True,
            batch_size=1000,
        )
        if len(dataset) == 0:
            raise ValueError(
                "No matching cells found for the specified group and sample IDs."
            )

    # Step 2: Filter based on presence or absence of cluster
    if isp_cluster_id is not None:
        if fixed_cluster_ids is None:
            fixed_cluster_ids = []

        if existed:
            # (1) isp_cluster type must be in cluster_types
            # (2) cluster_types must have more than one cluster type (excluding fixed clusters)
            token_dataset = dataset.filter(
                function=lambda batch: [
                    (isp_cluster_id in x and len(list((set(x) - set(fixed_cluster_ids)))) > 1)
                    for x in batch["cluster_types"]
                ],
                batched=True,
                batch_size=1000,
                num_proc=4,
            )

            # (3) OE/expansion of isp_cluster not exceed the maximum of TME cells
            if cluster_endpoint > 1:
                token_dataset = token_dataset.filter(
                    function=batch_exceed_total_cells_filter,
                    batched=True,
                    batch_size=1000,
                    num_proc=4,
                    fn_kwargs={
                        "isp_cluster": isp_cluster_id,
                        "endpoints": cluster_endpoint,
                    },
                )

            # Optional: Further filter by specific compositional interval
            if interval is not None and interval not in ([0, 1], (0, 1)):

                def compute_cluster_ratio(
                    example: Dict[str, Any],
                    isp_cluster_id: int,
                    fixed_cluster_ids: Optional[List[int]],
                ) -> Dict[str, float]:
                    cluster_types = example["cluster_types"]
                    if fixed_cluster_ids is not None:
                        cluster_types = [
                            tme_type
                            for tme_type in cluster_types
                            if tme_type not in fixed_cluster_ids
                        ]
                    counter = Counter(cluster_types)
                    return {
                        "ratio": counter[isp_cluster_id] / len(cluster_types)
                    }

                token_dataset = token_dataset.map(
                    compute_cluster_ratio,
                    fn_kwargs={
                        "isp_cluster_id": isp_cluster_id,
                        "fixed_cluster_ids": fixed_cluster_ids,
                    },
                )
                token_dataset = tme_interval_dataset(token_dataset, interval)

        else:
            # Keep only cells where the cluster is not present
            token_dataset = dataset.filter(
                function=lambda batch: [
                    isp_cluster_id not in x for x in batch["cluster_types"]
                ],
                batched=True,
                batch_size=1000,
                num_proc=4,
            )

    # Step 3: Sample a subset of the filtered dataset
    if ratio is not None:
        token_dataset = sample_dataset(token_dataset, ratio)

    return token_dataset


def filter_tme_rank_dataset(
    dataset: Dataset,
    isp_cluster_id: int,
    cell_cluster_dict: Dict[int, int],
    proj: str = "xenium",
    tokens: Optional[List[int]] = None,
    existed: bool = True,
    ratio: Optional[Union[int, float]] = None,
    work_dir: Optional[str] = None,
) -> Dataset:
    """
    Filter cell dataset whose tme_cells can be implemented with TME_Rank ISP.

    Args:
        dataset: HuggingFace Dataset to filter.
        isp_cluster_id: Target cluster ID.
        cell_cluster_dict: Mapping {cell_id: cluster_id}.
        proj: Project name (default: "xenium").
        tokens: Target gene tokens.
        existed: Whether to filter for presence or absence.
        ratio: Sampling ratio.
        work_dir: Working directory path.

    Returns:
        Filtered dataset.
    """
    # All cells that (1) belong to one cluster_id and (2) can express the target gene
    dataset_all = load_from_disk(
        work_dir + f"data/{proj}/datasets/{proj}.dataset"
    )
    clu_cell_ids = np.array(
        [
            cell_id
            for cell_id, cluster in cell_cluster_dict.items()
            if cluster == isp_cluster_id
        ]
    )
    dataset_all_clu = dataset_all.select(clu_cell_ids - 1)

    disable_caching()
    dataset_all_clu_filt = filter_token_dataset(
        dataset_all_clu, tokens=tokens, existed=True
    )
    expressed_cluster_cells = set(dataset_all_clu_filt["cell_id"])

    # Filter cell dataset whose tme_cells (not) overlap with above cells
    if existed:
        token_dataset = dataset.filter(
            function=lambda batch: [
                not (set(x)).isdisjoint(expressed_cluster_cells)
                for x in batch["tme_cells"]
            ],
            batched=True,
            batch_size=1000,
            num_proc=4,
        )
    else:
        token_dataset = dataset.filter(
            function=lambda batch: [
                (set(x)).isdisjoint(expressed_cluster_cells)
                for x in batch["tme_cells"]
            ],
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

    # Step 3: Sample a subset of the filtered dataset
    if ratio is not None:
        token_dataset = sample_dataset(token_dataset, ratio)

    return token_dataset


# =============================================================================
# Model and Dataset Loading
# =============================================================================


def load_model_and_dataset(
    model_id: str,
    task_dataset_dir: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Union[Any, Tuple[Any, Dataset]]:
    """
    Load a pre-trained model and its matched task dataset for similarity ISP.

    Args:
        model_id: Key for selecting the model path and dataset version from model_dict.
        task_dataset_dir: Base directory where datasets are stored.
        model_config: Configuration dict with keys:
            - model_dict: Dictionary mapping model_id to (path, version, type)
            - device: GPU device ID (e.g., 0 for 'cuda:0')

    Returns:
        If task_dataset_dir is None: model only
        Otherwise: (model, dataset) tuple

    Raises:
        ValueError: If model_id not found in model_dict.
        FileNotFoundError: If required dataset files not found.
    """
    # Check model_id validity
    model_dict = model_config["model_dict"]
    device = model_config["device"]

    if model_id not in model_dict:
        raise ValueError(f"Model ID '{model_id}' not found in model_dict.")

    model_path, data_version = model_dict[model_id][0], model_dict[model_id][1]

    # Load model
    model = pu.load_model(
        model_type="Pretrained_TME",
        num_classes=None,
        model_directory=model_path,
        mode="eval",
        quantize=False,
        device=f"cuda:{device}",
    )

    if task_dataset_dir is None:
        return model

    # Load dataset
    check_file_1 = f"{task_dataset_dir}/{data_version}/isp_from.dataset"
    check_file_2 = f"{task_dataset_dir}/{data_version}/isp.dataset"
    if os.path.exists(check_file_1):
        dataset_file = check_file_1
    elif os.path.exists(check_file_2):
        dataset_file = check_file_2
    else:
        raise FileNotFoundError(f"{check_file_1} or {check_file_2} not found")

    dataset = load_from_disk(dataset_file)

    # Rename to tme_cells and tme_types columns
    dataset = tu.modify_tme_dataset(model, dataset)

    return model, dataset


# =============================================================================
# ISP Score Calculation
# =============================================================================


def calc_relative_similarity(
    shifted_cell_emb: Union[List, torch.Tensor],
    shifted_offsets: List[float],
    goal_state_emb: Union[List, torch.Tensor],
) -> List[Tuple[float, float]]:
    """
    Calculate the relative similarity of each shifted perturbation with respect to the goal state.

    Args:
        shifted_cell_emb: Embedding vectors of the shifted cells (list or tensor).
        shifted_offsets: List of offsets corresponding to the shifted embeddings.
        goal_state_emb: Embedding of the goal state (list or tensor).

    Returns:
        List of tuples, each containing (offset, relative_similarity_to_original).

    Raises:
        ValueError: If offset 0 is not in sims2goal.
    """
    # Calculate cosine similarities between goal and shifted embeddings
    cos = torch.nn.CosineSimilarity(dim=0)
    goal_emb_tensor = torch.tensor(goal_state_emb)
    sims2goal = {
        offset: cos(goal_emb_tensor, shifted_cell_emb[idx]).item()
        for idx, offset in enumerate(shifted_offsets)
    }

    # Calculate relative similarity to the original (offset 0)
    if 0 not in sims2goal:
        raise ValueError("Error: (original) offset 0 must exist.")
    sims2goal_relative = [
        (offset, sim - sims2goal[0]) for offset, sim in sims2goal.items()
    ]

    return sims2goal_relative


def calc_isp_score(
    cell_id_dict: Dict[int, List[float]],
    score_type: str,
    score_method: str,
) -> float:
    """
    Calculate the ISP score based on in-silico perturbation results.

    Args:
        cell_id_dict: Dictionary where keys are perturbation offsets (int),
            and values are lists/tuples, e.g., {offset: [score]}.
        score_type: One of ["OE", "KD", "KO", "KI"].
        score_method: One of ["sum", "area", "endpoint"].

    Returns:
        Calculated ISP score (float).

    Raises:
        ValueError: If score_type or score_method is invalid, or if KO requires key -9999.
    """
    if score_type not in ["OE", "KD", "KO", "KI"]:
        raise ValueError("score_type must be 'OE' or 'KD' or 'KO' or 'KI'")
    if score_method not in ["sum", "area", "endpoint"]:
        raise ValueError("score_method must be 'sum' or 'area' or 'endpoint'")

    if score_type in ["OE", "KI"]:
        cell_id_dict_subset = {
            x: y for x, y in cell_id_dict.items() if x >= 0
        }
    elif score_type == "KD":
        cell_id_dict_subset = {
            x: y
            for x, y in cell_id_dict.items()
            if (x <= 0) and (x != -9999)
        }
    elif score_type == "KO":
        if -9999 not in cell_id_dict:
            raise ValueError("For KO, key -9999 must exist in cell_id_dict.")
        return cell_id_dict[-9999][0]

    if score_method == "sum":
        isp_score = sum([y[0] for x, y in cell_id_dict_subset.items()])

    elif score_method == "area":
        x_vals = np.abs(np.array(list(cell_id_dict_subset.keys())))
        y_vals = np.array([v[0] for v in cell_id_dict_subset.values()])
        sort_idx = np.argsort(x_vals)
        isp_score = np.trapz(y_vals[sort_idx], x_vals[sort_idx])

    elif score_method == "endpoint":
        # The key with the largest absolute value
        max_abs_key = max(cell_id_dict_subset, key=lambda k: abs(k))
        isp_score = cell_id_dict_subset[max_abs_key][0]

    return isp_score


# =============================================================================
# ISP Run Configuration Utilities
# =============================================================================


def isp_run_suffix(
    isp_run_config: Dict[str, Any],
    isp: str = "cell",
) -> str:
    """
    Generate a suffix string based on ISP run configuration.

    Args:
        isp_run_config: Configuration dict with keys 'window', 'endpoints', 'ko_method', 'ki_method'.
        isp: Type of ISP, either "cell" or "tme".

    Returns:
        Suffix string encoding the ISP configuration.

    Example:
        >>> config = {"window": 100, "endpoints": 1, "ko_method": None, "ki_method": None}
        >>> suffix = isp_run_suffix(config, isp="cell")
        >>> print(suffix)  # W100_EP1_KO0_KI0
    """
    if isp == "cell":
        # "target_rank": e.g. W100_EP1
        p1 = f"W{int(isp_run_config['window'])}"
        p2 = f"EP{int(isp_run_config['endpoints'])}"

    elif isp == "tme":
        # "tme_rank": e.g. W0_EP1
        if isp_run_config["window"] == 0 and isp_run_config["endpoints"] == 1:
            p1 = "W0"
            p2 = "EP1"

        # "tme_composition": e.g. Wn0.2_EP0.0, W0.2_EP2.0
        else:
            p1 = f"W{str(float(isp_run_config['window']))}".replace(
                "-", "n"
            )  # n: negative window
            p2 = f"EP{str(float(isp_run_config['endpoints']))}"

    # Knockout method
    p3 = f"KO1" if isp_run_config["ko_method"] is not None else "KO0"

    # Knock-in method
    p4 = f"KI1" if isp_run_config["ki_method"] is not None else "KI0"

    return f"{p1}_{p2}_{p3}_{p4}"


def generate_cell_isp_score_sets(
    isp_run_config: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """
    Generate score types and methods for target cell ISP based on configuration.

    Args:
        isp_run_config: ISP configuration dict.

    Returns:
        List of tuples (score_type, score_method).
    """
    isp_score_types = []

    if isp_run_config["ki_method"] is None:
        if isp_run_config["window"] != 0:
            isp_score_types.extend([("OE", "area"), ("KD", "area")])
        if isp_run_config["endpoints"]:
            isp_score_types.extend([("OE", "endpoint"), ("KD", "endpoint")])
        if isp_run_config["ko_method"] is not None:
            isp_score_types.append(("KO", "endpoint"))
    else:
        if isp_run_config["window"] != 0:
            isp_score_types.append(("KI", "area"))
        if isp_run_config["endpoints"]:
            isp_score_types.append(("KI", "endpoint"))

    return isp_score_types


def generate_tme_isp_score_sets(
    isp_run_config: Dict[str, Any],
    method: str,
) -> List[Tuple[str, str]]:
    """
    Generate score types and methods for TME ISP based on configuration.

    Args:
        isp_run_config: ISP configuration dict.
        method: Either "composition" or "rank".

    Returns:
        List of tuples (score_type, score_method).
    """
    if method == "composition":
        isp_score_types = []
        if isp_run_config["ki_method"] is None:
            if isp_run_config["window"] != 0:
                if isp_run_config["endpoints"] > 1:
                    isp_score_types.extend([("OE", "area")])
                else:
                    isp_score_types.extend([("KD", "area")])
            if isp_run_config["endpoints"] > 1:
                isp_score_types.extend([("OE", "endpoint")])
            else:
                isp_score_types.extend([("KD", "endpoint")])
    elif method == "rank":
        isp_score_types = [("OE", "endpoint"), ("KD", "endpoint")]

    return isp_score_types


# =============================================================================
# TME Dataset Version Utilities
# =============================================================================


def tme_dataset_versions(
    proj: str = "xenium",
    celldata: str = "epi",
    work_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    List available TME datasets for a given project and cell data type.

    Args:
        proj: Project name (default: "xenium").
        celldata: Cell data type, either "epi" or "all".
        work_dir: Working directory path.

    Returns:
        Dictionary mapping dataset versions to their paths.
    """
    dir_path = work_dir + f"data/{proj}/datasets/"
    if celldata == "epi":
        pattern = f"{proj}_epi_TME"
    elif celldata == "all":
        pattern = f"{proj}_TME"
    ds_names = sorted(
        [name for name in os.listdir(dir_path) if name.startswith(pattern)]
    )
    ds_versions = [name.split(".")[0].split("_")[-1] for name in ds_names]
    ds_paths = [dir_path + name for name in ds_names]
    tme_datasets = {
        ds_version: ds_path for ds_version, ds_path in zip(ds_versions, ds_paths)
    }
    return tme_datasets


def add_cluster_types(
    dataset: Dataset,
    isp_cluster_id: int,
    cell_cluster_dict: Dict[int, int],
) -> Dataset:
    """
    Add cluster_types column to dataset.

    Args:
        dataset: HuggingFace Dataset.
        isp_cluster_id: Target cluster ID.
        cell_cluster_dict: Mapping {cell_id: cluster_id}.

    Returns:
        Dataset with cluster_types column added.
    """
    if isp_cluster_id in range(1, 9):
        # For the main cell type id, directly copy
        dataset = dataset.add_column("cluster_types", dataset["tme_types"])
    else:
        # For novel cluster id, update the cluster types
        disable_caching()

        def make_cluster_types(example: Dict[str, Any]) -> Dict[str, Any]:
            example["cluster_types"] = [
                cell_cluster_dict[x] for x in example["tme_cells"]
            ]
            return example

        dataset = dataset.map(make_cluster_types)
        # enable_caching()

    return dataset